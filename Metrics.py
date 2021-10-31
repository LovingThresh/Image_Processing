from keras import backend as K
import tensorflow as tf
import numpy as np


# 自定义损失函数
def Asymmetry_Binary_Loss(y_true, y_pred):
    # 想要损失函数更加关心裂缝的标签值1
    y_true_0, y_pred_0 = y_true[:, :, :, 0] * 5, y_pred[:, :, :, 0] * 5
    # y_true_0, y_pred_0 = y_true[:, :, :, 0] * 255, y_pred[:, :, :, 0] * 255
    y_true_1, y_pred_1 = y_true[:, :, :, 1], y_pred[:, :, :, 1]
    mse = tf.losses.mean_squared_error

    return mse(y_true_0, y_pred_0) + mse(y_true_1, y_pred_1)


def Precision(y_true, y_pred):
    """精确率"""
    tp = K.sum(K.round(K.clip(y_true[:, :, :, 0], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision


def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true[:, :, :, 0], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))   # true positives
    pp = K.sum(K.round(K.clip(y_true[:, :, :, 0], 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall


def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1


def IOU(y_true: tf.Tensor,
        y_pred: tf.Tensor):
    predict = K.round(K.clip(y_pred[:, :, :, 0], 0, 1))
    Intersection = K.sum(y_true[:, :, :, 0] * predict)
    Union = K.sum(y_true[:, :, :, 0] + predict)
    iou = Intersection / (Union - Intersection)
    return iou