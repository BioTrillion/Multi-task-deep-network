import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataset import DatasetImageMaskContourDist
import glob
from models import UNet, UNet_DCAN, UNet_DMTN, PsiNet, UNet_ConvMCD
from tqdm import tqdm
import numpy as np
import cv2; import pandas as pd
from utils import create_validation_arg_parser
from enum import Enum
from circleFit import circle_fit_jacobian

PI = np.pi
TOL_OFFSET_TO_RADIUS = .05  # pixels between center of ellipse and circler
TOL_ECC_PUPIL = 1.1  # eccentricity, no units
TOL_ECC_IRIS = 1.05  # eccentricity, no units
TOL_THETA = 5  # degrees
DEG2RAD = np.pi / 180

class Anatomy(Enum):
    PUPIL = 1
    IRIS = 2

# def cuda(x):
# return x.cuda(async=True) if torch.cuda.is_available() else x


def build_model(model_type):

    if model_type == "unet":
        model = UNet(num_classes=2)
    if model_type == "dcan":
        model = UNet_DCAN(num_classes=2)
    if model_type == "dmtn":
        model = UNet_DMTN(num_classes=2)
    if model_type == "psinet":
        model = PsiNet(num_classes=3)
    if model_type == "convmcd":
        model = UNet_ConvMCD(num_classes=3)

    return model

def rotating_calipers(contour):
    """Estimate pupil or iris diameter using rotating calipers method

    Parameters
    ----------
        contour : {cv2.countour}
            Structure containing vertices that comprise a contour--outline--of a blob

    Returns
    ----------
        radius_avg : float
            Average of rotating calipers measurements about the perimeter
        radius_median : float
            Median of rotating calipers measurements about the perimeter
        radius_min : float
            Minimum of rotating calipers measurements about the perimeter
        radius_max : float
            Maximum of rotating calipers measurements about the perimeter
    """
    xy = []
    diam_hor = []
    diam_ver = []

    for vertex in contour:
        xy.append(vertex[0])
    xy = np.array(xy)

    if xy.size == 0:
        return 0

    x = xy[:, 0]
    y = xy[:, 1]

    # For the horizontal axis (major)
    # for angle in np.arange(0, PI, PI/36): # for circular 
    for angle in np.arange(-PI/18, +PI/18, PI/72):  # for elliptical (DB)
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)
        x_rot = cos_ang * x + sin_ang * y  # we need to take min and max of either x or y
        # y_rot = -sin_rad * x + cos_rad * y
        diam_hor.append(x_rot.max() - x_rot.min())

    diam_hor = np.array(diam_hor)
    radius_avg_hor = diam_hor.mean() / 2
    radius_median_hor = np.median(diam_hor) / 2
    radius_min_hor = diam_hor.min() / 2
    radius_max_hor = diam_hor.max() / 2

    # For the horizontal axis (minor)
    # for angle in np.arange(0, PI, PI/36): # for circular 
    for angle in np.arange(-PI/18 + PI/2, +PI/18 + PI/2, PI/72):  # for elliptical (DB)
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)
        x_rot = cos_ang * x + sin_ang * y  # we need to take min and max of either x or y
        # y_rot = -sin_rad * x + cos_rad * y
        diam_ver.append(x_rot.max() - x_rot.min())

    diam_ver = np.array(diam_ver)
    radius_avg_ver = diam_ver.mean() / 2
    radius_median_ver = np.median(diam_ver) / 2
    radius_min_ver = diam_ver.min() / 2
    radius_max_ver = diam_ver.max() / 2

    return radius_avg_hor, radius_avg_ver

def calculate_radius(pupil):
    """Additional ways to calculate pupil to iris radius ratio
    Parameters
    ----------
        pupil : {np.ndarray}
            pupil or iris pixel mask

    Returns
    ----------
        No return
    """

    pupil_radius_equiv = 0
    pupil_radius_min_enc_circle = 0
    pupil_radius_equiv_conv_hull = 0
    pupil_radius_rotcal_avg = 0
    pupil_radius_rotcal_med = 0
    pupil_radius_rotcal_min = 0
    pupil_radius_rotcal_max = 0

    radius_avg_hor = 0
    radius_median_hor = 0
    radius_avg_ver = 0
    radius_median_ver = 0

    # Radius as value of the circle with same area as blob
    if pupil.sum() > 0:
        pupil_blob_area_r = np.sqrt(pupil.sum()/np.pi)
        pupil_radius_area_based = pupil_blob_area_r

        # Based on blob properties
        pupil = 1 * pupil  # converts bool to int
        if pupil.dtype is np.dtype('int32'):
            pupil_thresh = pupil
        else:
            _, pupil_thresh = cv2.threshold(pupil, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(pupil_thresh.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 0.1:
                # pupil_radius_equiv = np.sqrt(cv2.contourArea(contour) / PI) # equiv diam = sqrt(4*CA/pi)
                # hull_idx = cv2.convexHull(contour, returnPoints=False)  # will return a contour
                # pupil_radius_min_enc_circle = cv2.minEnclosingCircle(contour)[1]
                # pupil_radius_equiv_conv_hull = np.sqrt(cv2.contourArea(contour[hull_idx.T[0]]) / PI)
                # pupil_radius_rotcal_avg, pupil_radius_rotcal_med, pupil_radius_rotcal_min, pupil_radius_rotcal_max = rotating_calipers(contour)
                radius_avg_hor, radius_avg_ver = rotating_calipers(contour)

    return radius_avg_hor, radius_avg_ver

def calc_elliptical_iou(ellipse_coords_1, ellipse_coords_2):
    """
    Calculates the IoU for 2 ellipses

    Args:
        ellipse_coords_1 (tuple): 1st tuple containing ellipse parameters
        ellipse_coords_2 (tuple): 2nd tuple containing ellipse parameters

    Returns:
        iou (float): Intersection-over-union (IoU) value for the 2 ellipses
    """

    (x1, y1, rx1, ry1, t1) = ellipse_coords_1
    (x2, y2, rx2, ry2, t2) = ellipse_coords_2

    img1 = np.zeros((500,500))
    img2 = img1.copy()

    img1 = cv2.ellipse(img1, (round(x1), round(y1)), (round(rx1), round(ry1)), t1, 0.0, 360.0, 255, -1)
    img2 = cv2.ellipse(img2, (round(x2), round(y2)), (round(rx2), round(ry2)), t2, 0.0, 360.0, 255, -1)

    img1 = img1 > 0
    img2 = img2 > 0

    iou = np.logical_and(img1, img2).sum() / np.logical_or(img1, img2).sum()

    return iou

def calculate_elliptical_radius(pred_blob, anatomy, img_num=-1, out_folder=None, org_fname=None):
    """Estimate pupil or iris diameter using rotating calipers method

    Parameters
    ----------
        pred_blob : {np.array}
            Predicted mask output of the NN model (blob-shaped, mostly circular/elliptical)
        anatomy : {string}
            Class to which the blob belongs to (Pupil/Iris)
    
    Returns
    ----------
        cX : int
            X co-ordinate of the center of the fitted ellipse
        cY : int
            Y co-ordinate of the center of the fitted ellipse
        rX : float
            Major axis of the fitted ellipse
        rY : float
            Minor axis of the fitted ellipse
        theta : float
            Theta angle of the fitted ellipse
        mom_cX : int
            X co-ordinate of the centroid of the predicted blob
        mom_cY : int
            Y co-ordinate of the centroid of the predicted blob
    """
    max_val = 1
    threshold = 0.5
    
    # cX, cY, rX, rY, theta = (0, 0, 0.0, 0.0, 0.0)
    cX, cY, rX, rY, theta, mom_cX, mom_cY = (0, 0, 0.0, 0.0, 0.0, 0, 0)

    if pred_blob.sum() > 0:
        pred_blob = 1 * pred_blob  # Converts bool to int

        if pred_blob.dtype is np.dtype('int32'):
            blob_thresh = pred_blob
        else:
            _, blob_thresh = cv2.threshold(pred_blob, threshold, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(blob_thresh.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour_area = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0.1 and len(contour) > 4:
                contour_area.append(area)

        if len(contour_area) > 0:
            idx_max = np.argmax(contour_area)
            lrg_contour = contours[idx_max]

            if cv2.contourArea(lrg_contour) > 0.1:
                pt_x = np.array([x[0][0] for x in lrg_contour])
                pt_y = np.array([x[0][1] for x in lrg_contour])
                circ_x, circ_y, circ_r = circle_fit_jacobian(pt_x, pt_y)
                try:
                    # rX, rY, cX, cY, theta = fit_ellipse(pt_x, pt_y)  # Theta is in radians from normal x-y coordinate system.

                    (e_x, e_y), (e_M, e_m), e_a = cv2.fitEllipse(lrg_contour)
                    # (e_x, e_y), (e_M, e_m), e_a = cv2.fitEllipseAMS(lrg_contour)
                    # (e_x, e_y), (e_M, e_m), e_a = cv2.fitEllipseDirect(lrg_contour)

                    cX = e_x; cY = e_y; rX = e_M/2; rY = e_m/2
                    theta = e_a #* DEG2RAD

                    center_offset = np.sqrt((circ_x - cX)**2 + (circ_y - cY)**2)
                    # theta = theta * 180 / PI  # Converting to degrees

                    # Compute the moments of a contour (to calculate Centroid later)
                    M = cv2.moments(lrg_contour)
                    mom_cX = int(M["m10"] / M["m00"])
                    mom_cY = int(M["m01"] / M["m00"])

                except:
                    center_offset = -1  # negative number to trigger a failure in validation
                    theta = 1e6         # large enough number to trigger a failure in validation

                # Determine whether to use circle or ellipse
                use_ellipse = True
                if rX is None or np.isnan(rX) or np.isnan(rY) or np.isnan(cX) or np.isnan(cY):
                    use_ellipse = False
                elif center_offset < 0 or center_offset > rX * TOL_OFFSET_TO_RADIUS:
                    use_ellipse = False
                elif anatomy == Anatomy.PUPIL and rX/rY > TOL_ECC_PUPIL:
                    use_ellipse = False
                elif anatomy == Anatomy.IRIS and rX/rY > TOL_ECC_IRIS:
                    use_ellipse = False
                elif anatomy == Anatomy.PUPIL and np.abs(np.abs(theta) - 90) > TOL_THETA:  # thetas are often negative. We want them close to 90 deg
                    use_ellipse = False

                # DEBUG Code
                if img_num != -1 and out_folder != None:
                    
                    blob_temp = np.ones((blob_thresh.shape[0], blob_thresh.shape[1], 3)) * 255
                    blob_temp[:,:,2] = blob_thresh
                    cv2.drawContours(blob_temp, lrg_contour, -1, (255, 0, 0), 1)

                    text_1 = ''; text_2 = ''; text_3 = ''
                    if not(rX is None or np.isnan(cX) or np.isnan(cY) or np.isnan(rX) or np.isnan(rY) or np.isnan(theta)):
                        blob_temp = cv2.ellipse(blob_temp, (int(cX), int(cY)), (int(rX), int(rY)), theta, 0.0, 360.0, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
                        text_1 = f'Ellipse: cx={cX:.2f}, cy={cY:.2f}, rx={rX:.2f}, ry={rY:.2f}'
                        
                    if not(circ_r is None or np.isnan(circ_x) or np.isnan(circ_y) or np.isnan(circ_r)):
                        blob_temp = cv2.circle(blob_temp, (int(circ_x), int(circ_y)), int(circ_r), color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
                        text_2 = f'Circle: cx={circ_x:.2f}, cy={circ_y:.2f}, r={circ_r:.2f}'

                    text_3 = f'Final: {"Ellipse" if use_ellipse else "Circle"}'

                    font_scale = 0.25
                    if text_1:
                        cv2.putText(blob_temp, text_1 , (5, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color=(0, 0, 0))
                    if text_2:
                        cv2.putText(blob_temp, text_2 , (5, 140), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color=(0, 0, 0))
                    if text_3:
                        cv2.putText(blob_temp, text_3 , (5, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color=(0, 0, 0))

                    if anatomy == Anatomy.PUPIL:
                        if org_fname is None:
                            cv2.imwrite(os.path.join(out_folder, f"{img_num}_ellipse_pupil.png"), blob_temp)
                        else:
                            cv2.imwrite(os.path.join(out_folder, f"{org_fname}_{img_num}_ellipse_pupil.png"), blob_temp)
                    elif anatomy == Anatomy.IRIS:
                        if org_fname is None:
                            cv2.imwrite(os.path.join(out_folder, f"{img_num}_ellipse_iris.png"), blob_temp)
                        else:
                            cv2.imwrite(os.path.join(out_folder, f"{org_fname}_{img_num}_ellipse_iris.png"), blob_temp)

                if not use_ellipse:
                    cX = circ_x; cY = circ_y; rX = circ_r; rY = circ_r; theta = 0.0

    return cX, cY, rX, rY, theta, mom_cX, mom_cY

if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()
    val_path = os.path.join(args.val_path, "*.png")
    model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    distance_type = "dist_contour"
    val_file_names = glob.glob(val_path)
    valLoader = DataLoader(DatasetImageMaskContourDist(val_file_names, distance_type))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = build_model(model_type)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    count = 0
    pe_p_x = 0; pe_p_y = 0; pe_i_x = 0; pe_i_y = 0
    iou_p = 0; iou_i = 0

    # for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(
    for i, (img_file_name, inputs) in enumerate(
        tqdm(valLoader)
    ):

        inputs = inputs.to(device)
        outputs1, outputs2, outputs3 = model(inputs)

        outputs1 = outputs1.detach().cpu().numpy().squeeze()
        outputs2 = outputs2.detach().cpu().numpy().squeeze()
        outputs3 = outputs3.detach().cpu().numpy().squeeze()

        res = np.zeros((256, 256))
        indices = np.argmax(outputs1, axis=0)
        res[indices == 2] = 100
        res[indices == 1] = 50
        res[indices == 0] = 0

        index_pupil = 1; index_iris = 2
        pupil_diam_x = max(np.where(indices == index_pupil)[0]) - min(np.where(indices == index_pupil)[0])
        pupil_diam_y = max(np.where(indices == index_pupil)[1]) - min(np.where(indices == index_pupil)[1])

        iris_diam_x = max(np.where(indices == index_iris)[0]) - min(np.where(indices == index_iris)[0])
        iris_diam_y = max(np.where(indices == index_iris)[1]) - min(np.where(indices == index_iris)[1])

        blank_mask = np.zeros(res.shape)
        blank_mask[indices == index_pupil] = 100
        pupil_mask = blank_mask

        blank_mask = np.zeros(res.shape)
        blank_mask[indices == index_iris] = 100
        iris_mask = blank_mask
        iris_mask += pupil_mask

        # Predicted radius using the rotating calipers method
        pupil_r_x, pupil_r_y = calculate_radius(pupil_mask)
        iris_r_x, iris_r_y = calculate_radius(iris_mask)

        # Read groundtruth CSV file
        gtruth_file = img_file_name[0].replace('image', 'groundtruth').replace('.png', '.csv')
        gtruth_df = pd.read_csv(gtruth_file)

        gtruth_p_r_x = float(gtruth_df['radiusX_p_true']); gtruth_p_r_y = float(gtruth_df['radiusY_p_true'])
        gtruth_i_r_x = float(gtruth_df['radiusX_i_true']); gtruth_i_r_y = float(gtruth_df['radiusY_i_true'])

        # Avg % error [ % error = (pred - gtruth) / gtruth * 100 ]
        pe_p_x += (abs(pupil_r_x - gtruth_p_r_x) / gtruth_p_r_x) * 100
        pe_p_y += (abs(pupil_r_y - gtruth_p_r_y) / gtruth_p_r_y) * 100
        pe_i_x += (abs(iris_r_x - gtruth_i_r_x) / gtruth_i_r_x) * 100
        pe_i_y += (abs(iris_r_y - gtruth_i_r_y) / gtruth_i_r_y) * 100

        # Centroid of the pupil/iris
        _, _, _, _, _, iris_centr_cX, iris_centr_cY  = calculate_elliptical_radius(iris_mask, Anatomy.IRIS)
        _, _, _, _, _, pupil_centr_cX, pupil_centr_cY  = calculate_elliptical_radius(pupil_mask, Anatomy.PUPIL)

        # Calculating the IoU
        ellParams_p_true = (float(gtruth_df["coord_p_true_x"]), float(gtruth_df["coord_p_true_y"]), float(gtruth_df["radiusX_p_true"]), 
                            float(gtruth_df["radiusY_p_true"]), float(gtruth_df["theta_p_true"]))
        ellParams_p_pred = (pupil_centr_cX, pupil_centr_cY, pupil_r_x, pupil_r_y, 0)
        iou_p_ellipse = calc_elliptical_iou(ellParams_p_true, ellParams_p_pred)
        iou_p += iou_p_ellipse

        ellParams_i_true = (float(gtruth_df["coord_i_true_x"]), float(gtruth_df["coord_i_true_y"]), float(gtruth_df["radiusX_i_true"]), 
                            float(gtruth_df["radiusY_i_true"]), float(gtruth_df["theta_i_true"]))
        ellParams_i_pred = (iris_centr_cX, iris_centr_cY, iris_r_x, iris_r_y, 0)
        iou_i_ellipse = calc_elliptical_iou(ellParams_i_true, ellParams_i_pred)
        iou_i += iou_i_ellipse

        count += 1

        # print(f"{img_file_name}, {pupil_diam_x}, {pupil_diam_y}, {iris_diam_x} {iris_diam_y}")
        print(f"{img_file_name[0]}, {pupil_r_x * 2}, {pupil_r_y * 2}, {iris_r_x * 2} {iris_r_y * 2}")                   # Predicted
        print(f"{img_file_name[0]}, {gtruth_p_r_x * 2}, {gtruth_p_r_y * 2}, {gtruth_i_r_x * 2} {gtruth_i_r_y * 2}")     # Groundtruth
        print(f"{img_file_name[0]}, iou_p: {iou_p_ellipse}, iou_i: {iou_i_ellipse}")
        print("\n")

        output_path = os.path.join(
            save_path, "mask_" + os.path.basename(img_file_name[0])
        )
        cv2.imwrite(output_path, res)

    print(f"Avg % Error (Pupil - Radius X): {100 - pe_p_x / count}")
    print(f"Avg % Error (Pupil - Radius Y): {100 - pe_p_y / count}")
    print(f"Avg % Error (Iris - Radius X): {100 - pe_i_x / count}")
    print(f"Avg % Error (Iris - Radius Y): {100 - pe_i_y / count}")

    print(f"Pupil IoU: {iou_p / count}")
    print(f"Iris IoU: {iou_i / count}")
