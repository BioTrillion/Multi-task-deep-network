# base_path="E:\\Data\\training_set\\iPhone_BT_12thMarch21_80_split"
train_path='E:\\Data\\training_set\\iPhone_BT_12thMarch21_80_split\\image' 
val_path='E:\\Data\\val_set\\iPhone_BT_12thMarch21_10_split\\image' 
model_type='convmcd'
object_type='polyp'
save_path='E:\\Source\\Multi-task-deep-network\\models'
python train.py --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} --object_type ${object_type} --save_path ${save_path}
