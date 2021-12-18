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

# from keras_flops import get_flops
import I_data
import Metrics
import pylib as py
from Callback import CheckpointSaver, EarlyStopping, CheckpointPlot, DynamicLearningRate
from Metrics import *
from I_data import *
import module

from SegementationModels import *
from tensorflow.keras import models
import matplotlib.pyplot as plt

from Compare.SegNet import segnet
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ----------------------------------------------------------------------
#                               parameter
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Stage_1')
parser.add_argument('--datasets_dir', default=r'Stage_1')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--load_size', type=int, default=512)
parser.add_argument('--crop_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=10)

parser.add_argument('--loss', default='binary_crossentropy loss')
parser.add_argument('--loss_parameter', default='1')

parser.add_argument('--model', default='ResNet')
parser.add_argument('--Student_model', default='False')
parser.add_argument('--Student_model_Convolution', default='Standard Convolution')

parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
parser.add_argument('--Illustrate', default=' with Attention with No ShallowConnect with DataArgumentation'
                                            ' No Knowledge Distillation'
                                            ' Res-Net SaveModel'
                                            ' 在Pad函数做了适应性调整，以适应TensorRT'
                                            ' StandardConv2D + 1'
                                            ' 实验结果 - 18！！'
                                            ' 使用binary_crossentropy_loss'
                                            ' Stage_1数据集,使用数据增强函数, 然后用binary_crossentropy_loss')
args = parser.parse_args()

# ----------------------------------------------------------------------
#                               dataset
# ----------------------------------------------------------------------

# lines, num_train, num_val = get_data()
# batch_size = 10
# train_dataset = get_dataset_label(lines[:num_train], batch_size)
# validation_dataset = get_dataset_label(lines[num_train:], batch_size)

train_lines, num_train = get_data(path=r'L:\ALASegmentationNets\Data\Stage_1\train.txt', training=False)
validation_lines, num_val = get_data(path=r'L:\ALASegmentationNets\Data\Stage_1\val.txt', training=False)
batch_size = 1
train_dataset = get_dataset_label(train_lines, batch_size,
                                  A_img_paths=r'L:\ALASegmentationNets\Data\Stage_1\train\img/',
                                  B_img_paths=r'L:\ALASegmentationNets\Data\Stage_1\train\mask/',
                                  C_img_paths=r'C:\Users\liuye\Desktop\data\train_2\teacher_mask/',
                                  shuffle=True,
                                  KD=False)
validation_dataset = get_dataset_label(validation_lines, batch_size,
                                       A_img_paths=r'L:\ALASegmentationNets\Data\Stage_1\val\img/',
                                       B_img_paths=r'L:\ALASegmentationNets\Data\Stage_1\val\mask/',
                                       C_img_paths=r'C:\Users\liuye\Desktop\data\val\teacher_mask/',
                                       shuffle=True,
                                       KD=False)


def ChangeAsGeneratorFunction(x):
    return lambda: (data for data in x)


train_data = ChangeAsGeneratorFunction(train_dataset)
validation_data = ChangeAsGeneratorFunction(validation_dataset)

# 将普通的生成器变成Dataset
keras_train_dataset = tf.data.Dataset.from_generator(train_data, output_types=np.float32)
keras_validation_dataset = tf.data.Dataset.from_generator(validation_data, output_types=np.float32)

keras_train_dataset = keras_train_dataset.map(I_data.map_function_for_keras,
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size) \
    .prefetch(tf.data.experimental.AUTOTUNE)

# ----------------------------------------------------------------------
#                               model
# ----------------------------------------------------------------------

# model = module.ResnetGenerator_with_ThreeChannel((512, 512, 3), attention=True, ShallowConnect=False, dim=32)
# model = module.StudentNet(attention=True)
model = module.U_Net(512, 512)
# Encoder = resnet34(512, 512, 2)
# model = ResNetDecoder(Encoder, 2)
# model = keras.models.load_model(r'C:\Users\liuye\Desktop\ep026-val_loss2101.036',
#                                 custom_objects={'M_Precision': M_Precision,
#                                                 'M_Recall': M_Recall,
#                                                 'M_F1': M_F1,
#                                                 'M_IOU': M_IOU,
#                                                 'mean_iou_keras': mean_iou_keras,
#                                                 'A_IOU': A_IOU,
#                                                 # 'H_KD_Loss': H_KD_Loss,
#                                                 # 'S_KD_Loss': S_KD_Loss,
#                                                 'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss
#                                                 }
#                                 )
# model = segnet((512, 512), 2)
model.summary()
# flops = get_flops(model)
# print(f"FLOPS: {flops / 10 ** 9:.03} G")
# initial_learning_rate = 2e-6
initial_learning_rate = 5e-5

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
    os.makedirs(r'E:/output/{}'.format(c))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='E:/output/{}/tensorboard/'.format(c))
    checkpoint = tf.keras.callbacks.ModelCheckpoint('E:/output/{}/checkpoint/'.format(c) +
                                                    'ep{epoch:03d}-val_loss{val_loss:.3f}/',
                                                    # 'Output_Label_loss:.3f}-val_acc{'
                                                    # 'Output_Label_accuracy:.3f}/',
                                                    monitor='val_accuracy', verbose=0,
                                                    save_best_only=False, save_weights_only=False,
                                                    mode='auto', period=1)

    os.makedirs(r'E:/output/{}/plot/'.format(c))
    plot_path = r'E:/output/{}/plot/'.format(c)
    checkpoints_directory = r'E:/output/{}/checkpoints/'.format(c)
    checkpointplot = CheckpointPlot(generator=validation_dataset, path=plot_path)
    checkpoints = tf.train.Checkpoint()
    manager = tf.train.CheckpointManager(checkpoints, directory=os.path.join(checkpoints_directory, "ckpt"),
                                         max_to_keep=3)
    checkpoints = CheckpointSaver(manager=manager)
    py.args_to_yaml('E:/output/{}/settings.yml'.format(c), args)

# ----------------------------------------------------------------------
#                               train
# ----------------------------------------------------------------------
    model.compile(optimizer=optimizer,
                  loss=Metrics.Asymmetry_Binary_Loss,
                  metrics=['accuracy', M_Precision, M_Recall, M_F1, M_IOU, mean_iou_keras, A_IOU])

    if training:
        model.fit(train_dataset,
                  steps_per_epoch=max(1, num_train // batch_size),
                  epochs=args.epoch,
                  validation_data=validation_dataset,
                  validation_steps=max(1, num_val // batch_size),
                  initial_epoch=0,
                  callbacks=[tensorboard, checkpoint, checkpoints, EarlyStopping, checkpointplot, DynamicLearningRate])

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
        model = module.StudentNet(dim=32, n_blocks=4, attention=True, Separable_convolution=False)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)

        model.compile(optimizer=optimizer,
                      loss={'Output_Label': Metrics.H_KD_Loss, 'Soft_Label': Metrics.S_KD_Loss},
                      metrics=['accuracy', A_Precision, A_Recall, A_F1, A_IOU])

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
test = False
out_tensorflow_lite = False
out_tensorRT_model = False
plot_predict = False
plot_mask = False
if test:
    test_path = r'L:\ALASegmentationNets\Data\Stage_1\val.txt'
    test_lines, num_test = get_data(test_path, training=False)
    batch_size = 1
    A_test_img_paths = r'L:\ALASegmentationNets\Data\Stage_1\val\img/'
    B_test_img_paths = r'L:\ALASegmentationNets\Data\Stage_1\val\mask/'
    C_test_img_paths = r'C:\Users\liuye\Desktop\data\val\teacher_mask/'
    test_dataset_label = get_test_dataset_label(test_lines, A_test_img_paths, B_test_img_paths,
                                                KD=False)
    model = keras.models.load_model(r'C:\Users\liuye\Desktop\ep125-val_loss1040.307',
                                    custom_objects={'Precision': A_Precision,
                                                    'Recall': A_Recall,
                                                    'F1': A_F1,
                                                    'IOU': A_IOU,
                                                    # 'H_KD_Loss': H_KD_Loss,
                                                    # 'S_KD_Loss': S_KD_Loss,
                                                    'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss
                                                    })
    model.compile(optimizer=optimizer,
                  loss=Metrics.Asymmetry_Binary_Loss,
                  metrics=['accuracy', A_Precision, A_Recall, A_F1, A_IOU, Asymmetry_Binary_Loss])
    model.evaluate(test_dataset_label[0], test_dataset_label[1], batch_size=1)
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
