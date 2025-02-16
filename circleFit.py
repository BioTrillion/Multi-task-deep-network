import scipy
from scipy      import  odr
from numpy.ma import row_stack
import numpy as np

# from https://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle.html


# == METHOD 3b ==
method_3b  = "odr with jacobian"

def f_3b(beta, x):
    """ implicit definition of the circle """
    return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2

def jacb(beta, x):
    """ Jacobian function with respect to the parameters beta.
    return df_3b/dbeta
    """
    xc, yc, r = beta
    xi, yi    = x

    df_db    = np.empty((beta.size, x.shape[1]))
    df_db[0] =  2*(xc-xi)                     # d_f/dxc
    df_db[1] =  2*(yc-yi)                     # d_f/dyc
    df_db[2] = -2*r                           # d_f/dr

    return df_db

def jacd(beta, x):
    """ Jacobian function with respect to the input x.
    return df_3b/dx
    """
    xc, yc, r = beta
    xi, yi    = x

    df_dx    = np.empty_like(x)
    df_dx[0] =  2*(xi-xc)                     # d_f/dxi
    df_dx[1] =  2*(yi-yc)                     # d_f/dyi

    return df_dx

def calc_estimate(data):
    """ Return a first estimation on the parameter from the data  """
    xc0, yc0 = data.x.mean(axis=1)
    r0 = np.sqrt((data.x[0]-xc0)**2 +(data.x[1] -yc0)**2).mean()
    return xc0, yc0, r0

# def calc_R(xc, yc):
#     """ calculate the distance of each data points from the center (xc, yc) """
#     return np.sqrt((x-xc)**2 + (y-yc)**2)


# for implicit function :
#       data.x contains both coordinates of the points
#       data.y is the dimensionality of the response

# x=[596.038, 607.924, 674.151, 594.34 , 606.226, 672.453]
# y=[490.755, 548.491, 502.642, 487.358, 548.491, 504.34 ]

def circle_fit_jacobian(x, y):
    lsc_data  = odr.Data(row_stack([x, y]), y=1)
    lsc_model = odr.Model(f_3b, implicit=True, estimate=calc_estimate, fjacd=jacd, fjacb=jacb)
    lsc_odr   = odr.ODR(lsc_data, lsc_model)    # beta0 has been replaced by an estimate function
    lsc_odr.set_job(deriv=3)                    # use user derivatives function without checking
    # lsc_odr.set_iprint(iter=1, iter_step=1)     # print details for each iteration
    lsc_out   = lsc_odr.run()

    xc_3b, yc_3b, R_3b = lsc_out.beta
    # Ri_3b       = calc_R(xc_3b, yc_3b)
    # residu_3b   = sum((Ri_3b - R_3b)**2)

    return xc_3b, yc_3b, R_3b