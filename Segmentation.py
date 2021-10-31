# !/usr/bin/envs/tensorflow python
# -*- coding:utf-8 -*-
# @Time : 2021/8/26 17:30
# @Author : Ye
# @File : Segmentation.py
# @SoftWare : Pycharm
import argparse
import datetime
import time

import cv2

import Metrics
import pylib as py
from Metrics import *
from I_data import *
from Callback import *
import module
from plot import plot_heatmap
from tensorflow.keras import models
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# ----------------------------------------------------------------------
#                               parameter
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='HEYE')
parser.add_argument('--datasets_dir', default='HEYE_img')
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--load_size', type=int, default=512)
parser.add_argument('--crop_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--loss', default='my losses mse')
parser.add_argument('--model', default='ReSNet')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
parser.add_argument('--Illustrate', default=' Define My Losses with Attention'
                                            ' Knowledge Distillation')
args = parser.parse_args()

# ----------------------------------------------------------------------
#                               dataset
# ----------------------------------------------------------------------

# lines, num_train, num_val = get_data()
# batch_size = 10
# train_dataset = get_dataset_label(lines[:num_train], batch_size)
# validation_dataset = get_dataset_label(lines[num_train:], batch_size)

train_lines, num_train = get_data(path=r'train_HEYE.txt', training=False)
validation_lines, num_val = get_data(path=r'validation_HEYE.txt', training=False)
batch_size = 1
train_dataset = get_dataset_label(train_lines, batch_size,
                                  A_img_paths=r'C:\Users\liuye\Desktop\data\train\img/',
                                  B_img_paths=r'C:\Users\liuye\Desktop\data\train\mask/',
                                  C_img_paths=r'C:\Users\liuye\Desktop\data\train\teacher_mask/',
                                  size=(512, 512),
                                  shuffle=True,
                                  KD=False)
validation_dataset = get_dataset_label(validation_lines, batch_size,
                                       A_img_paths=r'C:\Users\liuye\Desktop\data\val\img/',
                                       B_img_paths=r'C:\Users\liuye\Desktop\data\val\mask/',
                                       C_img_paths=r'C:\Users\liuye\Desktop\data\val\teacher_mask/',
                                       size=(512, 512),
                                       shuffle=True,
                                       KD=False)

# ----------------------------------------------------------------------
#                               model
# ----------------------------------------------------------------------

model = module.ResnetGenerator(attention=True)
# model = module.StudentNet(attention=True)
# model = module.U_Net(227, 227)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)

# ----------------------------------------------------------------------
#                               output
# ----------------------------------------------------------------------
training = False
KD = False

if training or KD:
    a = str(datetime.datetime.now())
    b = list(a)
    b[10] = '-'
    b[13] = '-'
    b[16] = '-'
    c = ''.join(b)
    os.makedirs(r'./output/{}'.format(c))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./output/{}/tensorboard/'.format(c))
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./output/{}/checkpoint/'.format(c) +
                                                    'ep{epoch:03d}-val_loss{'
                                                    'Output_Label_loss:.3f}-val_acc{'
                                                    'Output_Label_accuracy:.3f}.h5',
                                                    monitor='val_accuracy', verbose=0,
                                                    save_best_only=False, save_weights_only=False,
                                                    mode='auto', period=1)
    checkpoints_directory = r'./output/{}/checkpoints/'.format(c)

    checkpoints = tf.train.Checkpoint()
    manager = tf.train.CheckpointManager(checkpoints, directory=os.path.join(checkpoints_directory, "ckpt"),
                                         max_to_keep=3)
    checkpoints = CheckpointSaver(manager=manager)
    py.args_to_yaml('./output/{}/settings.yml'.format(c), args)

# ----------------------------------------------------------------------
#                               train
# ----------------------------------------------------------------------
model.compile(optimizer=optimizer,
              loss=Metrics.Asymmetry_Binary_Loss,
              metrics=['accuracy', Precision, Recall, F1, IOU])
if training:
    model.fit(train_dataset,
              steps_per_epoch=max(1, num_train // batch_size),
              epochs=args.epoch,
              validation_data=validation_dataset,
              validation_steps=max(1, num_val // batch_size),
              initial_epoch=0,
              callbacks=[tensorboard, checkpoint, checkpoints])

# ---------------------------------------------------------------------
#                       Knowledge Distillation
# ----------------------------------------------------------------------

if KD:
    train_dataset = get_dataset_label(train_lines, batch_size,
                                      A_img_paths=r'C:\Users\liuye\Desktop\data\train\img/',
                                      B_img_paths=r'C:\Users\liuye\Desktop\data\train\mask/',
                                      C_img_paths=r'C:\Users\liuye\Desktop\data\train\teacher_mask/',
                                      size=(512, 512),
                                      shuffle=True,
                                      KD=True)
    validation_dataset = get_dataset_label(validation_lines, batch_size,
                                           A_img_paths=r'C:\Users\liuye\Desktop\data\val\img/',
                                           B_img_paths=r'C:\Users\liuye\Desktop\data\val\mask/',
                                           C_img_paths=r'C:\Users\liuye\Desktop\data\val\teacher_mask/',
                                           size=(512, 512),
                                           shuffle=True,
                                           KD=True)
    model = module.StudentNet(attention=True)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)

    model.compile(optimizer=optimizer,
                  loss={'Output_Label': Metrics.H_KD_Loss, 'Soft_Label': Metrics.S_KD_Loss},
                  metrics=['accuracy', Precision, Recall, F1, IOU])

    model.fit_generator(train_dataset,
                        steps_per_epoch=max(1, num_train // batch_size),
                        epochs=args.epoch,
                        validation_data=validation_dataset,
                        validation_steps=max(1, num_val // batch_size),
                        initial_epoch=0,
                        callbacks=[tensorboard, checkpoint, checkpoints])

# ---------------------------------------------------------------------
#                               test
# ----------------------------------------------------------------------
test = True
out_tensorflow_lite = False
plot_predict = True
plot_mask = False
if test:
    test_path = r'I:\Image Processing\validation_HEYE.txt'
    test_lines, num_test = get_data(test_path, training=False)
    batch_size = 1
    A_test_img_paths = r'C:\Users\liuye\Desktop\data\val\img/'
    B_test_img_paths = r'C:\Users\liuye\Desktop\data\val\mask/'
    C_test_img_paths = r'C:\Users\liuye\Desktop\data\val\teacher_mask/'
    test_dataset_label = get_test_dataset_label(test_lines, A_test_img_paths, B_test_img_paths, C_test_img_paths, KD=True)
    model = keras.models.load_model(r'I:\Image Processing\output\2021-10-31-14-51-02.725162\checkpoint\ep299'
                                    r'-val_loss0.226-val_acc0.947.h5',
                                    custom_objects={'Precision': Precision,
                                                    'Recall': Recall,
                                                    'F1': F1,
                                                    'IOU': IOU,
                                                    'H_KD_Loss': H_KD_Loss,
                                                    'S_KD_Loss': S_KD_Loss
                                                    })
    model.compile(optimizer=optimizer,
                  loss=Metrics.Asymmetry_Binary_Loss,
                  metrics=['accuracy', Precision, Recall, F1, IOU, H_KD_Loss, S_KD_Loss])

    # 尝试输出TensorFlow Lite模型
    if out_tensorflow_lite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

        # Save the model
        with open('student_model_float16.tflite', 'wb') as f:
            f.write(tflite_model)

    # 输出模型预测结果
    if plot_predict:
        aa = tf.convert_to_tensor(test_dataset_label[0])
        a = time.time()
        # model.evaluate(test_dataset_label[0], test_dataset_label[1], batch_size=batch_size)
        model.predict(aa, batch_size=1)
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
        # plot_heatmap(predict)

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
