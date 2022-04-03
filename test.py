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

model = keras.models.load_model(r'E:\MCFF_checkpoint\ep083-val_loss5790.019',
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

image_path = r'D:\MCFF\Image/'
image_list = os.listdir(image_path)
for image in image_list:
    img_array = cv2.imread(image_path + image)
    img_array = img_array / 255.0  # 标准化
    img_array = img_array * 2 - 1
    img_array = img_array.reshape(1, 448, 448, 3)
    predict = model(img_array)[-1]
    predict = predict.numpy()
    predict = (predict[:, :, :, 0] > 0.4).astype(np.uint8).reshape(448, 448, 1) * 255
    predict = np.concatenate([predict, predict, predict], axis=2)

    cv2.imwrite(image_path + 'SegNet_' + image, predict)
