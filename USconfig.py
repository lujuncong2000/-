#-*- codeing = utf-8 -*-
#@Time : 2022/3/30 20:31
#@Author : 鹿俊聪
#@File : USconfig.py
#@Software : PyCharm
width_mult=1.0
width_mult_list=[0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
width_mult_range=[0.35, 1.0]
reset_parameters=True
num_sample_training=4
recal_batch=100
# cumulative_bn_stats=False
lr_scheduler='linear_decaying'
num_epochs=60
lr_warmup_epochs=0