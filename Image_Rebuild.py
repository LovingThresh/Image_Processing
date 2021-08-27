# 主要是使用对其进行重建

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2
import os

import module
import tf2lib as tl
import imlib as im
from Image_Progessing import get_file_path

G_A2B = module.ResnetGenerator(input_shape=(227, 227, 3))
G_B2A = module.ResnetGenerator(input_shape=(227, 227, 3))

D_A = module.ConvDiscriminator(input_shape=(227, 227, 3))
D_B = module.ConvDiscriminator(input_shape=(227, 227, 3))

G_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
D_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           r'G:\学习\CycleGAN-Tensorflow-2-master\output\crack\checkpoints/',
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

checkpoint.restore(r'G:\学习\CycleGAN-Tensorflow-2-master\output\crack\checkpoints\ckpt-13.index')


def Image_rebuild(image_path):
    B = cv2.imread(image_path)
    B = B.reshape((1, 227, 227, 3))
    img = tf.image.resize(B, [227, 227])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(
    # img, crop_size)
    img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
    img = img * 2 - 1
    A2B = G_A2B(img, training=False)
    A2B2A = G_B2A(A2B, training=False)
    # B2A2B = G_A2B(B2A, training=False)
    # B2A2B2A = G_B2A(B2A2B, training=False)
    img = im.immerge(np.concatenate([A2B2A], axis=0), n_rows=1)
    if not os.path.exists(r'./Rebuild_Image_95'):
        os.makedirs(r'./Rebuild_Image_95')
    re_path = r'./Rebuild_Image_95/{}.jpg'.format(image_path[-13:-8])
    im.imwrite(img, re_path)


# rebuilt Image
path = './Mix_img/95/image/'
file_list = get_file_path(path)
for i in file_list:
    Image_rebuild(i)
