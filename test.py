# !/usr/bin/envs/tensorflow python
# -*- coding:utf-8 -*-
# @Time : 2021/8/26 17:30
# @Author : Ye
# @File : Segmentation.py
# @SoftWare : Pycharm
import argparse
import datetime
import time
import os
import shutil

# from keras_flops import get_flops
# from Student_model import student_model
import keras.models
import numpy as np
import tensorflow as tf
import Metrics
import pylib as py
from Callback import CheckpointSaver, EarlyStopping, CheckpointPlot, DynamicLearningRate
from Metrics import *
from I_data import *
import module
from SegementationModels import *
from model_profiler import model_profiler
from tensorflow.keras import models
import matplotlib.pyplot as plt

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
#
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model = keras.models.load_model(r'E:\output\2022-03-29-21-31-54.123277\checkpoint\ep001-val_loss0.223',
                                custom_objects={'M_Precision': M_Precision,
                                                'M_Recall': M_Recall,
                                                'M_F1': M_F1,
                                                'M_IOU': M_IOU,
                                                'A_Precision': A_Precision,
                                                'A_Recall': A_Recall,
                                                'A_F1': A_F1,
                                                # 'mean_iou_keras': mean_iou_keras,
                                                'A_IOU': A_IOU,
                                                # 'H_KD_Loss': H_KD_Loss,
                                                # 'S_KD_Loss': S_KD_Loss,
                                                'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss,
                                                # 'DilatedConv2D': Layer.DilatedConv2D,
                                                }
                                )

model.evaluate(validation_dataset, steps=250)
model.evaluate(test_dataset, steps=250)

image = tf.random.normal((8, 448, 448, 3))
image = tf.convert_to_tensor(image, dtype=tf.float32)
for i in range(30):
    model.predict(image)
a = time.time()
for i in range(125):
    model.predict(image)
b = time.time()
print((b - a))

image_path = r'E:\MCFF_checkpoint\Image_/'
result_path = r'E:\MCFF_checkpoint\Image/'
image_list = os.listdir(image_path)
for image in image_list:
    img_array = cv2.imread(image_path + image)
    img_array = img_array / 255.0  # 标准化
    img_array = img_array * 2 - 1
    img_array = img_array.reshape(1, 448, 448, 3)
    predict = model(img_array)[-1]
    predict = predict.numpy()
    predict = predict.reshape((1, 448, 448, 2))
    predict = (predict[:, :, :, 0] > 0.4).astype(np.uint8).reshape(448, 448, 1) * 255
    predict = np.concatenate([predict, predict, predict], axis=2)

    cv2.imwrite(result_path + 'SegNet_' + image, predict)

times = [19.55, 21.85, 17.16, 36.00, 40.30, 36.65, 17.95, 13.94, 15.76, 18.74, 13.75, 18.35, 15.85, 16.54, 16.48]
flops = [2.25, 2.91, 1.89, 2.71, 2.70, 1.48, 3.44, 1.29, 0.84, 0.74, 1.70, 0.8, 0.95, 0.72, 0.65]
flops = [i * 200 for i in flops]
acc = [75.94, 70.11, 67.69, 74.10, 73.26, 77.03, 77.46, 72.56, 73.63, 70.29, 65.02, 70.60, 71.40, 78.22, 76.30]
colors = ['hotpink', 'hotpink', 'hotpink', 'hotpink', 'hotpink', 'hotpink', 'hotpink', '#88c999', '#88c999', '#88c999'
            , '#88c999', '#88c999', '#88c999', 'red', 'red']

plt.scatter(times, acc, s=flops, c=colors, alpha=0.5)
txt = ['UNet', 'SegNet', 'FCN', 'PAN', 'PSPNet', 'HRNet', 'ConvNext', 'MobileNet V1', 'MobileNet V2',
       'ESPNet', 'BiSegNet', 'GhostNet', 'EfficientNet', 'MCL', 'MCLB']

xy = [(times[0] + 1, acc[0] - 0.5), (times[1] + 1, acc[1] - 0.5), (times[2] + 1, acc[2] - 0.5), (times[3] + 1, acc[3] - 0.5),
      (times[4] - 1.8, acc[4] - 1.1), (times[5], acc[5] - 1), (times[6] + 1, acc[6]), (times[7] + 1, acc[7]),
      (times[8] + 1, acc[8]), (times[9] - 1, acc[9] - 1), (times[10] + 1, acc[10]), (times[11] - 4.5, acc[11] - 0.5),
      (times[12] + 0.5, acc[12]), (times[13] - 2.2, acc[13]), (times[14] - 2.7, acc[14])]

for i in range(len(times)):
    plt.annotate(txt[i], xy = xy[i])
plt.savefig('aaabbb.png', dpi=900)
