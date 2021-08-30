# !/usr/bin/envs/tensorflow python
# -*- coding:utf-8 -*-
# @Time : 2021/8/26 17:30
# @Author : Ye
# @File : Segmentation.py
# @SoftWare : Pycharm

import argparse
import datetime

import tensorflow.keras as keras
import keras.models

import pylib as py
from Metrics import *
from I_data import *
from Callback import *
import module
from plot import plot_heatmap

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# ----------------------------------------------------------------------
#                               parameter
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='95')
parser.add_argument('--datasets_dir', default='Mix_img')
parser.add_argument('--load_size', type=int, default=227)
parser.add_argument('--crop_size', type=int, default=227)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()

# ----------------------------------------------------------------------
#                               dataset
# ----------------------------------------------------------------------

lines, num_train, num_val = get_data()
batch_size = 10
train_dataset = get_dataset_label(lines[:num_train], batch_size)
validation_dataset = get_dataset_label(lines[num_train:], batch_size)

# ----------------------------------------------------------------------
#                               model
# ----------------------------------------------------------------------

model = module.ResnetGenerator()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)

# ----------------------------------------------------------------------
#                               output
# ----------------------------------------------------------------------

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
                                                'val_loss:.3f}-val_acc{'
                                                'val_accuracy:.3f}.h5',
                                                monitor='val_accuracy', verbose=0,
                                                save_best_only=False, save_weights_only=False,
                                                mode='auto', period=1)
checkpoints_directory = r'./output/{}/checkpoints/'.format(c)
checkpoints = CheckpointSaver(checkpoints_directory)

py.args_to_yaml('./output/{}/settings.yml'.format(c), args)

# ----------------------------------------------------------------------
#                               train
# ----------------------------------------------------------------------
training = False
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', Precision, Recall, F1, IOU])
if training:
    model.fit(train_dataset,
              steps_per_epoch=max(1, num_train // batch_size),
              epochs=50,
              validation_data=validation_dataset,
              validation_steps=max(1, num_val // batch_size),
              initial_epoch=0,
              callbacks=[tensorboard, checkpoint, checkpoints])

# ---------------------------------------------------------------------
#                               test
# ----------------------------------------------------------------------
test_path = r'I:\Image Processing\text.txt'
test_lines, num_test = get_data(test_path, training=False)
batch_size = 5
A_test_img_paths = r'I:\Image Processing\Test_Image\images/'
B_test_img_paths = r'I:\Image Processing\Test_Image\outputs\attachments/'
test_dataset_label = get_test_dataset_label(test_lines, A_test_img_paths, B_test_img_paths)
model = keras.models.load_model(r'output/2021-08-27-16-08-21.971270/checkpoint/ep049-val_loss0.030-val_acc0.993.h5',
                                custom_objects={'Precision': Precision,
                                                'Recall': Recall,
                                                'F1': F1,
                                                'IOU': IOU})
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', Precision, Recall, F1, IOU])
model.evaluate(test_dataset_label[0], test_dataset_label[1], batch_size=batch_size)
predict = model.predict(test_dataset_label[0][0].reshape(1, 227, 227, 3))
plot_heatmap(predict)
