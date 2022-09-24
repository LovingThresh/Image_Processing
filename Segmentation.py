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

import cv2
import numpy as np

import utils.layers
from builders.model_builder import builder

# from keras_flops import get_flops
# from Student_model import student_model
# import keras.models

import module
import Metrics
import pylib as py
import model_profiler

from Metrics import *
from I_data import *
from models.HRNet import seg_fc_hrnet
from models.ESPNet import ESPNet_tf
from models.ConvNext import ConvNext
from models.GhostNet import GhostModel
from models.EfficienttNet import efficient_net
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import losses
import matplotlib.pyplot as plt
from Callback import CheckpointSaver, EarlyStopping, CheckpointPlot, DynamicLearningRate

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ----------------------------------------------------------------------
#                               parameter
# ----------------------------------------------------------------------

# parser = argparse.ArgumentParser(description='Train a CAM model')
# parser.add_argument('dataset', type=str, help='train dataset config')
# parser.add_argument('dataset_type', type=str, help='train dataset config')
# parser.add_argument('size', type=int, help='train dataset config')
#
# args = parser.parse_args()


# data_path = args.dataset
# data_type = args.dataset_type
# size = (args.size, args.size)
size = (224, 224)
data_path = 'crack'
data_type = 'train_Positive_CAM_mask'
# ----------------------------------------------------------------------
#                               dataset
# ----------------------------------------------------------------------

# lines, num_train, num_val = get_data()
# batch_size = 10
# train_dataset = get_dataset_label(lines[:num_train], batch_size)
# validation_dataset = get_dataset_label(lines[num_train:], batch_size)

train_lines, num_train = get_data(
    path=r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\{}\train.txt'.format(data_path), training=False)
validation_lines, num_val = get_data(
    path=r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\{}\val.txt'.format(data_path), training=False)

# train_lines, num_train = get_data(path=r'L:\CRACK500\train.txt', training=False)
# validation_lines, num_val = get_data(path=r'L:\CRACK500\val.txt', training=False)
# test_lines, num_test = get_data(path=r'L:\CRACK500\test.txt', training=False)

batch_size = 1
epoch = 10
# 下面的代码适用于测试的
# -------------------------------------------------------------
# train_lines, num_train = train_lines[:2], 2
# validation_lines, num_val = validation_lines[:2], 2
# -------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------
#                                        非Teacher训练
# ---------------------------------------------------------------------------------------------------
train_dataset = get_dataset_label(train_lines, batch_size,
                                  A_img_paths=r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\{}\train_Positive/'.format(
                                      data_path),
                                  B_img_paths=r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\{}\ann_dir/{}/'.format(
                                      data_path, data_type),
                                  shuffle=True,
                                  KD=False,
                                  training=True,
                                  Augmentation=True)
validation_dataset = get_dataset_label(validation_lines, batch_size,
                                       A_img_paths=r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\{}\val_Positive/'.format(
                                           data_path),
                                       B_img_paths=r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\{}\ann_dir\val_true/'.format(
                                           data_path),
                                       shuffle=False,
                                       KD=False,
                                       training=False,
                                       Augmentation=False)
a = next(train_dataset)
b = next(validation_dataset)
# train_dataset = get_dataset_label(train_lines, batch_size,
#                                   A_img_paths=r'L:\CRACK500\traincrop/',
#                                   B_img_paths=r'L:\CRACK500\traincrop/',
#                                   shuffle=True,
#                                   KD=False,
#                                   training=True,
#                                   Augmentation=True)
# validation_dataset = get_dataset_label(validation_lines, batch_size,
#                                        A_img_paths=r'L:\CRACK500\valcrop/',
#                                        B_img_paths=r'L:\CRACK500\valcrop/',
#                                        shuffle=False,
#                                        KD=False,
#                                        training=False,
#                                        Augmentation=False)
#
# test_dataset = get_dataset_label(test_lines, batch_size,
#                                  A_img_paths=r'L:\CRACK500\testcrop/',
#                                  B_img_paths=r'L:\CRACK500\testcrop/',
#                                  shuffle=False,
#                                  KD=False,
#                                  training=False,
#                                  Augmentation=False)

# ---------------------------------------------------------------------------------------------------
#                                        Teacher训练
# ---------------------------------------------------------------------------------------------------
# 当temperature设置为0时，train_dataset不对标签做处理，即real_mix的值域是1~0
# train_dataset = get_teacher_dataset_label(train_lines,
#                                           A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\img/',
#                                           B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\mask/',
#                                           h_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_h\label/',
#                                           x_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_x\label/',
#                                           y_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_y\label/',
#                                           mix_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_mix\label/',
#                                           batch_size=batch_size,
#                                           shuffle=True,
#                                           temperature=0
#                                           )
#
# validation_dataset = get_teacher_dataset_label(validation_lines,
#                                                A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\img/',
#                                                B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\mask/',
#                                                h_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\teacher_mask\teacher_label_h\label/',
#                                                x_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\teacher_mask\teacher_label_x\label/',
#                                                y_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\teacher_mask\teacher_label_y\label/',
#                                                mix_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\teacher_mask\teacher_label_mix\label/',
#                                                batch_size=batch_size,
#                                                shuffle=False,
#                                                temperature=0,
#
#                                                )
#
# test_dataset = get_teacher_dataset_label(test_lines,
#                                          A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\img/',
#                                          B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\mask/',
#                                          h_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\teacher_mask\teacher_label_h\label/',
#                                          x_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\teacher_mask\teacher_label_x\label/',
#                                          y_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\teacher_mask\teacher_label_y\label/',
#                                          mix_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\teacher_mask\teacher_label_mix\label/',
#                                          batch_size=batch_size,
#                                          shuffle=False,
#                                          temperature=0
#                                          )

# def ChangeAsGeneratorFunction(x):
#     return lambda: (data for data in x)
#
#
# train_data = ChangeAsGeneratorFunction(train_dataset)
# validation_data = ChangeAsGeneratorFunction(validation_dataset)
#
# # 将普通的生成器变成Dataset
# keras_train_dataset = tf.data.Dataset.from_generator(train_data, output_types=np.float32)
# keras_validation_dataset = tf.data.Dataset.from_generator(validation_data, output_types=np.float32)
#
# keras_train_dataset = keras_train_dataset.map(I_data.map_function_for_keras,
#                                               num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size) \
#     .prefetch(tf.data.experimental.AUTOTUNE)

# ----------------------------------------------------------------------
#                               model
# ----------------------------------------------------------------------
temperature = 10
# 设置一个纯净版的ResnetGenerator_with_ThreeChannel，目前temperature对train_dataset不起作用，要与之相对应
# 纯净版包括哪些条件——普通卷积、无注意力机制、损失函数为平衡状态、KD方式为温度升降同时
# 条件均满足————可开始消融实验
# 消融实验-1-纯净版+注意力机制+不平衡损失函数+普通蒸馏（200改10）
# model = module.ResnetGenerator_with_ThreeChannel((448, 448, 3), output_channels=3, attention=True, ShallowConnect=False, dim=64,
#                                                  n_blocks=8,
#                                                  StudentNet=False, Temperature=temperature)
# ——————————————————————————————————————新测试——————————————————————————————————————

model, base_model = builder(2, size, model='UNet')

# model = seg_fc_hrnet(448, 448, channel=3, classes=2)

# model = ConvNext()

# 轻量化网络


# model, base_model = builder(2, (448, 448), model='UNet', base_model='MobileNetV1')
# model, base_model = builder(2, (448, 448), model='UNet', base_model='MobileNetV2')
# model = ESPNet_tf(classes=128, p=4, q=6)()
# model  = efficient_net()
# model = GhostModel(2, 448, 3).model
# ——————————————————————————————————————新测试——————————————————————————————————————


# model.summary()
# # model, base_model = builder(2, input_size=(448, 448), model='DenseASPP', base_model='DenseNet201')
batch_size = 1
profile = model_profiler.model_profiler(model, batch_size)
print(profile)
batch_size = 1
# flops = get_flops(model)
# print(f"FLOPS: {flops / 10 ** 9:.03} G")
# model = module.StudentNet(attention=True)

# model, base_model = builder(2, (448, 448), model='FCN-32s', base_model='DenseNet121')
# Encoder = resnet101(448, 448, 2)
# model = ResNetDecoder(Encoder, 2)
# 模型验证阶段

# model = module.U_Net(448, 448)

# 模型参数
# model.summary()

# batch_size = 4
# profile = model_profiler(model, batch_size)
#
# print(profile)

# model = module.ResnetGenerator_with_ThreeChannel(attention=True, ShallowConnect=False, dim=16, n_blocks=4)

#
model = keras.models.load_model(r'C:\Users\liuye\Desktop\ep083-val_loss5790.019',
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
# input = model.input
# output = model.layers[-1].input
# output = tf.math.softmax(output)
# model = keras.models.Model(inputs=input, outputs=[output, output, output, output, output])
# model.evaluate(validation_dataset, steps=250)
# model.evaluate(test_dataset, steps=250)
# model = segnet((512, 512), 2)
# model.summary()
initial_learning_rate = 5e-5
# initial_learning_rate = 5e-5
# initial_learning_rate_list = [1e-5, 5e-6, 2e-6, 1e-6]

# ---------------------------------------------------------------------------
#                              KD
# ---------------------------------------------------------------------------
#
# def teacher_model(Encoder, Temperature):
#     input_layer = Encoder.input
#     h = Encoder.layers[303].input
#     print(h.name)
#
#     x = Encoder.layers[304].input
#     print(x.name)
#
#     y = Encoder.layers[305].input
#     print(y.name)
#
#     mix = Encoder.layers[306].input
#     print(mix.name)
#
#     h = h / Temperature
#     x = x / Temperature
#     y = y / Temperature
#     mix = mix / Temperature
#
#     h = keras.layers.Softmax(name='Label_h_with_Temperature')(h)
#     x = keras.layers.Softmax(name='Label_x_with_Temperature')(x)
#     y = keras.layers.Softmax(name='Label_y_with_Temperature')(y)
#     mix = keras.layers.Softmax(name='Label_mix_with_Temperature')(mix)
#
#     Teacher_model = keras.models.Model(inputs=input_layer, outputs=[h, x, y, mix])
#     Teacher_model.trainable = False
#
#     return Teacher_model
#
#
# model = teacher_model(model, temperature)
# print(model.trainable)
# if model.trainable:
#     model.trainable = False
#
# img_path: str = r'L:\ALASegmentationNets_v2\Data\Stage_4\train\img/'
#
# img_name_list = os.listdir(img_path)
# for img in img_name_list:
#     path = img_path + img
#
#     tensor = cv2.imread(path)
#     tensor = tensor / 255.0
#     tensor = tensor * 2 - 1
#     tensor = np.reshape(tensor, (1, tensor.shape[0], tensor.shape[1], tensor.shape[2]))
#     predict = model.predict(tensor)
#     plt.imsave(r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_h\label/'
#                + img[:-4] + '.png', np.repeat(predict[0][0, :, :, 0:1], 3, axis=-1))
#     plt.imsave(r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_x\label/'
#                + img[:-4] + '.png', np.repeat(predict[1][0, :, :, 0:1], 3, axis=-1))
#     plt.imsave(r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_y\label/'
#                + img[:-4] + '.png', np.repeat(predict[2][0, :, :, 0:1], 3, axis=-1))
#     plt.imsave(r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_mix\label/'
#                + img[:-4] + '.png', np.repeat(predict[3][0, :, :, 0:1], 3, axis=-1))
#
optimizer = keras.optimizers.RMSprop(initial_learning_rate)
# optimizer = keras.optimizers.SGD(0.01, momentum=0.9, decay=0.0005)
# ----------------------------------------------------------------------
#                               output
# ----------------------------------------------------------------------
training = True
KD = False

if training or KD:
    a = str(datetime.datetime.now())
    b = list(a)
    b[10] = '-'
    b[13] = '-'
    b[16] = '-'
    c = ''.join(b)
    os.makedirs(r'E/output/{}'.format(c))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='E/output/{}/tensorboard/'.format(c))
    checkpoint = tf.keras.callbacks.ModelCheckpoint('E/output/{}/checkpoint/'.format(c) +
                                                    'ep{epoch:03d}-val_loss{val_loss:.3f}/',
                                                    # 'Output_Label_loss:.3f}-val_acc{'
                                                    # 'Output_Label_accuracy:.3f}/',
                                                    monitor='val_accuracy', verbose=0,
                                                    save_best_only=False, save_weights_only=False,
                                                    mode='auto', period=1)

    os.makedirs(r'E/output/{}/plot/'.format(c))
    plot_path = r'E/output/{}/plot/'.format(c)
    checkpoints_directory = r'E/output/{}/checkpoints/'.format(c)
    checkpointplot = CheckpointPlot(generator=validation_dataset, path=plot_path)
    checkpoints = tf.train.Checkpoint()
    manager = tf.train.CheckpointManager(checkpoints, directory=os.path.join(checkpoints_directory, "ckpt"),
                                         max_to_keep=3)
    checkpoints = CheckpointSaver(manager=manager)
    output_dir = r'E/output/{}'.format(c)
    module_dir = py.join(output_dir, 'module_code')
    py.mkdir(module_dir)

    # 本地复制源代码，便于复现(模型文件、数据文件、训练文件、测试文件)
    # 冷代码
    shutil.copytree('imlib', '{}/{}'.format(module_dir, 'imlib'))
    shutil.copytree('pylib', '{}/{}'.format(module_dir, 'pylib'))
    shutil.copytree('tf2lib', '{}/{}'.format(module_dir, 'tf2lib'))

    # 个人热代码
    shutil.copy('module.py', module_dir)
    shutil.copy('I_data.py', module_dir)
    shutil.copy('Metrics.py', module_dir)
    shutil.copy('Segmentation.py', module_dir)

    # ----------------------------------------------------------------------
    #                               train
    # ----------------------------------------------------------------------
    # loss = utils.losses.miou_loss()
    model.compile(optimizer=optimizer,
                  loss=Metrics.Asymmetry_Binary_Loss,
                  # loss=loss,
                  # {
                  # 'Label_h': Metrics.S_KD_Loss,
                  # 'Label_x': Metrics.S_KD_Loss,
                  # 'Label_y': Metrics.S_KD_Loss,
                  # 'Label_mix': Metrics.S_KD_Loss,
                  # 'Label_mix_for_real': Metrics.H_KD_Loss,
                  #       },
                  # Metrics.Asymmetry_Binary_Loss,

                  metrics=['accuracy', A_IOU, A_Precision, A_Recall])

    if training:
        model.fit(train_dataset,
                  steps_per_epoch=max(1, num_train // batch_size),
                  epochs=epoch,
                  validation_data=validation_dataset,
                  validation_steps=max(1, num_val // batch_size),
                  initial_epoch=0,
                  callbacks=[tensorboard, checkpoint, checkpoints, EarlyStopping, checkpointplot,
                             DynamicLearningRate])

    # ---------------------------------------------------------------------
    #                       Knowledge Distillation
    # ----------------------------------------------------------------------

    if KD:
        train_dataset = get_dataset_label(train_lines, batch_size,
                                          A_img_paths=r'C:\Users\liuye\Desktop\data\train\img/',
                                          B_img_paths=r'C:\Users\liuye\Desktop\data\train\mask/',
                                          C_img_paths=r'C:\Users\liuye\Desktop\data\train\teacher_mask/',
                                          shuffle=True,
                                          KD=True)
        validation_dataset = get_dataset_label(validation_lines, batch_size,
                                               A_img_paths=r'C:\Users\liuye\Desktop\data\val\img/',
                                               B_img_paths=r'C:\Users\liuye\Desktop\data\val\mask/',
                                               C_img_paths=r'C:\Users\liuye\Desktop\data\val\teacher_mask/',
                                               shuffle=True,
                                               KD=True)
        model = module.StudentNet(dim=32, n_blocks=4, attention=True, Separable_convolution=False)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)

        model.compile(optimizer=optimizer,
                      loss={'Output_Label': Metrics.H_KD_Loss, 'Soft_Label': Metrics.S_KD_Loss},
                      metrics=['accuracy', A_Precision, A_Recall, A_F1, A_IOU])

        model.fit_generator(train_dataset,
                            steps_per_epoch=200,
                            epochs=epoch,
                            validation_data=validation_dataset,
                            validation_steps=200,
                            initial_epoch=0,
                            callbacks=[tensorboard, checkpoint, checkpoints])

# ---------------------------------------------------------------------
#                               test
# ----------------------------------------------------------------------
test = False
out_tensorflow_lite = False
out_tensorRT_model = False
plot_predict = False
plot_mask = False
if test:
    test_path = r'L:\ALASegmentationNets\Data\Stage_4\val.txt'
    test_lines, num_test = get_data(test_path, training=False)
    batch_size = 1
    A_test_img_paths = r'L:\ALASegmentationNets\Data\Stage_4\val\img/'
    B_test_img_paths = r'L:\ALASegmentationNets\Data\Stage_4\val\mask/'
    C_test_img_paths = r'C:\Users\liuye\Desktop\data\val\teacher_mask/'
    test_dataset_label = get_test_dataset_label(test_lines, A_test_img_paths, B_test_img_paths,
                                                KD=False)
    initial_learning_rate = 5e-5
    optimizer = keras.optimizers.RMSprop(initial_learning_rate)
    model = keras.models.load_model(r'E:\output\2022-03-06-23-18-41.346776_SOTA4\checkpoint\ep025-val_loss2001.124',
                                    custom_objects={
                                        'M_Precision': M_Precision,
                                        'M_Recall': M_Recall,
                                        'M_F1': M_F1,
                                        'M_IOU': M_IOU,
                                        'mean_iou_keras': mean_iou_keras,
                                        'A_Precision': A_Precision,
                                        'A_Recall': A_Recall,
                                        'A_F1': A_F1,
                                        'A_IOU': A_IOU,
                                        'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss
                                    })
    model.compile(optimizer=optimizer,
                  loss=Metrics.Asymmetry_Binary_Loss,
                  metrics=['accuracy', A_Precision, A_Recall, A_F1, A_IOU, Asymmetry_Binary_Loss])
    model.evaluate(validation_dataset, steps=1000)
    # 尝试输出TensorFlow Lite模型
    if out_tensorflow_lite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        tflite_model = converter.convert()

        # Save the model
        with open('student_epp599.tflite', 'wb') as f:
            f.write(tflite_model)

    if out_tensorRT_model:
        params = tf.experimental.tensorrt.ConversionParams(
            precision_mode='FP32', maximum_cached_engines=16)

        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=r'output/2021-10-31-20-15-01.247650/checkpoint/ep529-val_loss0.213-val_acc0.946',
            conversion_params=params)
        converter.convert()
        converter.save(r'checkpoint/ep529-val_loss0.213-val_acc0.946')

    # 输出模型预测结果
    if plot_predict:
        aa = tf.convert_to_tensor(test_dataset_label[0])
        a = time.time()
        # model.evaluate(test_dataset_label[0], test_dataset_label[1], batch_size=batch_size)
        predict = model.predict(aa, batch_size=1)
        b = time.time()
        print(b - a)
        # a = test_dataset_label[0][0].reshape(1, 512, 512, 3)
        # start = datetime.datetime.now()
        # start = time.time()
        # predict = model.predict(test_dataset_label[0])
        # end = time.time()
        # end = datetime.datetime.now()
        # t = end - start
        # print(t)
        # plot_heatmap(predict[0][0, :, :, :])

    # 输出模型中的Mask
    if plot_mask:
        # 本次实验中使用到的mask是layer的[84]
        Mask_out = model.layers[84].output
        attention_mask_model = models.Model(inputs=model.input, outputs=model.layers[84].output)
        predict_img = test_dataset_label[0][0].reshape(1, 512, 512, 3)
        predict_img = tf.convert_to_tensor(predict_img)
        mask = attention_mask_model.predict(predict_img)
        plt.imshow(mask.reshape(512, 512))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.show()

# model = module.ResnetGenerator_with_ThreeChannel(attention=True, ShallowConnect=False, dim=32)
# flops = get_flops(model)
# print(f"FLOPS: {flops / 10 ** 9:.03} G")
# model = module.StudentNet(attention=True)
# model = module.U_Net(512, 512)
# Encoder = resnet34(512, 512, 2)
# model = ResNetDecoder(Encoder, 2)
# model = module.ResnetGenerator_with_ThreeChannel(attention=True, ShallowConnect=False, dim=32)

# 模型测试
# model = keras.models.load_model(r'C:\Users\liuye\Desktop\期末\课题\student_1_1',
#                                 custom_objects={'M_Precision': M_Precision,
#                                                 'M_Recall': M_Recall,
#                                                 'M_F1': M_F1,
#                                                 'M_IOU': M_IOU,
#                                                 'mean_iou_keras': mean_iou_keras,
#                                                 'A_Precision': A_Precision,
#                                                 'A_Recall': A_Recall,
#                                                 'A_F1': A_F1,
#                                                 'A_IOU': A_IOU,
#                                                 'H_KD_Loss': H_KD_Loss,
#                                                 'S_KD_Loss': S_KD_Loss,
#                                                 # 'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss,
#                                                 # 'DilatedConv2D': DilatedConv2D,
#                                                 }
#                                 )
# model.evaluate(validation_dataset, steps=250)
# model.evaluate(test_dataset, steps=250)


# model = keras.models.Model(inputs=model.input, outputs=model.output[-1])
# model.save(r'C:\Users\liuye\Desktop\student_1_1/')
# model.save_weights(r'C:\Users\liuye\Desktop\student_1_1_weights/')
# model = segnet((512, 512), 2)
# model.summary()
# initial_learning_rate = 3e-6
# initial_learning_rate = 2e-6

# ---------------------------------------------------------------------------
#                              KD
# ---------------------------------------------------------------------------
# def teacher_model(Encoder, Temperature):
#     input_layer = Encoder.input
#     h = Encoder.layers[303].input
#     print(h.name)
#
#     x = Encoder.layers[304].input
#     print(x.name)
#
#     y = Encoder.layers[305].input
#     print(y.name)
#
#     mix = Encoder.layers[306].input
#     print(mix.name)
#
#     h = h / Temperature
#     x = x / Temperature
#     y = y / Temperature
#     mix = mix / Temperature
#
#     h = keras.layers.Softmax(name='Label_h_with_Temperature')(h)
#     x = keras.layers.Softmax(name='Label_x_with_Temperature')(x)
#     y = keras.layers.Softmax(name='Label_y_with_Temperature')(y)
#     mix = keras.layers.Softmax(name='Label_mix_with_Temperature')(mix)
#
#     Teacher_model = keras.models.Model(inputs=input_layer, outputs=[h, x, y, mix])
#     Teacher_model.trainable = False
#
#     return Teacher_model
#
#
# model = teacher_model(model, temperature)
# print(model.trainable)
# if model.trainable:
#     model.trainable = False
#
# img_path: str = r'L:\ALASegmentationNets\Data\Stage_4\test\img/'
#
# img_name_list = os.listdir(img_path)
# for img in img_name_list:
#     path = img_path + img
#
#     tensor = cv2.imread(path)
#     tensor = tensor / 255.0
#     tensor = tensor * 2 - 1
#     tensor = np.reshape(tensor, (1, tensor.shape[0], tensor.shape[1], tensor.shape[2]))
#     predict = model.predict(tensor)
#     plt.imsave(r'L:\ALASegmentationNets\Data\Stage_4\test\teacher_mask\teacher_label_h\label/'
#                + img[:-4] + '.png', np.repeat(predict[0][0, :, :, 0:1], 3, axis=-1))
#     plt.imsave(r'L:\ALASegmentationNets\Data\Stage_4\test\teacher_mask\teacher_label_x\label/'
#                + img[:-4] + '.png', np.repeat(predict[1][0, :, :, 0:1], 3, axis=-1))
#     plt.imsave(r'L:\ALASegmentationNets\Data\Stage_4\test\teacher_mask\teacher_label_y\label/'
#                + img[:-4] + '.png', np.repeat(predict[2][0, :, :, 0:1], 3, axis=-1))
#     plt.imsave(r'L:\ALASegmentationNets\Data\Stage_4\test\teacher_mask\teacher_label_mix\label/'
#                + img[:-4] + '.png', np.repeat(predict[3][0, :, :, 0:1], 3, axis=-1))

# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train = tf.reshape(x_train, (60000, 28, 28, 1))
# x_test = tf.reshape(x_test, (10000, 28, 28, 1))
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False),
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)


# filepath = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\crack\Positive/'
# output_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\label\crack\Positive_0.2/'
# # os.mkdir(output_path)
# #
# files = os.listdir(filepath)
# for i in files:
#     image = cv2.imread(filepath + i)
#     image = cv2.resize(image, (448, 448))
#     image = image / 255.
#     image = image * 2 - 1
#
#     predict = model.predict(image.reshape(1, 448, 448, 3))
#     predict = cv2.resize(predict[-1].reshape(448, 448, 2), (224, 224))
#     predict = (predict[:, :, 0] > 0.2).astype(np.uint16) * 255
#     cv2.imwrite(output_path + i, predict)
# img_path = r'T:\Data_liu\data_third\img_dir\val/'
# save_path = r'T:\Data_liu\data_third\img_dir/'
# file_list = os.listdir(img_path)
# with open(save_path + 'val.txt', 'w') as f:
#     for file in file_list:
#         f.write(file + '\n')


for h, i in enumerate(validation_dataset):
    if h > 999:
        break
    prediction = model.predict(i[0])
    prediction = prediction[:, :, :, 1].reshape((224, 224))
    prediction = np.uint8(prediction > 0.5)
    cv2.imwrite('prediction_output_MCFF_0.5/' + validation_lines[h].split(',')[0][:-4] + '.png', prediction)


def A_Precision(y_true, y_pred):
    """精确率"""
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)

    tp = K.sum(
        K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # predicted positives
    precision = (tp + 1e-8) / (pp + 1e-8)
    return precision


def A_Recall(y_true, y_pred):
    """召回率"""
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)
    tp = K.sum(
        K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)))  # possible positives

    recall = (tp + 1e-8) / (pp + 1e-8)
    return recall


def A_F1(y_true, y_pred):
    """F1-score"""
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)
    precision = A_Precision(y_true, y_pred)
    recall = A_Recall(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def A_IOU(y_true: tf.Tensor,
          y_pred: tf.Tensor):
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)
    predict = K.round(K.clip(y_pred[:, :, :, 0], 0, 1))
    Intersection = K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) * predict)
    Union = K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) + predict)
    iou = (Intersection + 1e-8) / (Union - Intersection + 1e-8)
    return iou


def A_AC(y_true: tf.Tensor,
         y_pred: tf.Tensor):
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)
    predict = K.round(K.clip(y_pred[:, :, :, 0], 0, 1))
    Intersection = K.sum(tf.cast(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) == predict, dtype=tf.float32))
    Union = 224 * 224
    iou = (Intersection + 1e-8) / (Union + 1e-8)
    return iou


o_img_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\crack\val_Positive/'
p_img_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\crack\ann_dir\val_true/'
save_path = r'prediction_output_CRFs_0.5/'

A_iou_list = np.array([])
A_pr_list = np.array([])
A_re_list = np.array([])
A_f1_list = np.array([])
A_ac_list = np.array([])

for file in os.listdir(o_img_path):
    img_path_ = o_img_path + file
    p_img_path_ = p_img_path + file[:-4] + '.png'
    save_path_ = save_path + file[:-4] + '.png'
    prediction = cv2.imread(save_path_, cv2.IMREAD_GRAYSCALE)
    true = cv2.imread(p_img_path_, cv2.IMREAD_GRAYSCALE)
    prediction = prediction.astype(np.float32)
    true = true.astype(np.float32)
    prediction = tf.convert_to_tensor(prediction.reshape((1, 1, 224, 224)))
    true = tf.convert_to_tensor(true.reshape((1, 1, 224, 224)))
    iou = A_IOU(true, prediction)
    pr = A_Precision(true, prediction)
    re = A_Recall(true, prediction)
    f1 = A_F1(true, prediction)
    AC = tf.keras.metrics.Accuracy(
        name='accuracy', dtype=None)
    ac = AC(true, prediction)
    A_iou_list = np.append(A_iou_list, iou.numpy())
    A_pr_list = np.append(A_pr_list, pr.numpy())
    A_re_list = np.append(A_re_list, re.numpy())
    A_f1_list = np.append(A_f1_list, f1.numpy())
    A_ac_list = np.append(A_ac_list, ac.numpy())
np.mean(A_iou_list)
np.mean(A_pr_list)
np.mean(A_re_list)
np.mean(A_f1_list)
np.mean(A_ac_list)
