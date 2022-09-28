# -*- coding: utf-8 -*-
# @Time    : 2022/7/2 16:56
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : ConvNext.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras as keras


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ConvNext(input_shape=(448, 448, 3), output_channel=2, dim=96,
                                   n_downsampling=2, n_ResBlock=9,
                                   norm='layer_norm', attention=False):
    Norm = _get_norm_layer(norm)
    n_upsampling = n_downsampling
    init_dim = dim

    def ResNeXt_block(x):
        dim = x.shape[-1]
        h = x
        x = keras.layers.DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', use_bias=False)(x)
        x = Norm()(x)
        x = keras.layers.Conv2D(filters=dim * 4, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = keras.activations.gelu(x)
        x = keras.layers.Conv2D(filters=dim, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)

        x = keras.layers.Add()([x, h])

        return x

    h = inputs = keras.Input(shape=input_shape)
    h = keras.layers.Conv2D(filters=dim, kernel_size=(4, 4), strides=(4, 4), padding='valid', use_bias=False)(h)
    for i in range(n_downsampling):
        dim = dim * 2
        for _ in range(3):
            h = ResNeXt_block(h)
        h = keras.layers.Conv2D(filters=dim, kernel_size=(2, 2), strides=(2, 2), padding='valid', use_bias=False)(h)

    assert h.shape[-1] == dim

    for i in range(n_ResBlock):
        h = ResNeXt_block(h)

    for i in range(n_upsampling):
        dim = dim / 2
        for _ in range(3):
            h = ResNeXt_block(h)
        h = keras.layers.Conv2DTranspose(filters=dim, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                         use_bias=False)(h)

    assert h.shape[-1] == init_dim

    h = keras.layers.Conv2DTranspose(filters=init_dim, kernel_size=(4, 4), strides=(4, 4), padding='valid',
                                     use_bias=False)(h)

    h = keras.layers.Conv2D(filters=output_channel, kernel_size=(7, 7), strides=(1, 1), padding='same',
                            use_bias=False)(h)
    h = keras.layers.Softmax()(h)
    return keras.Model(inputs=inputs, outputs=h)
