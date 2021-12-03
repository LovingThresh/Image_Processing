from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


# 自定义损失函数
def Asymmetry_Binary_Loss(y_true, y_pred):
    # 想要损失函数更加关心裂缝的标签值1
    y_true_0, y_pred_0 = y_true[:, :, :, 0] * 5, y_pred[:, :, :, 0] * 5
    # y_true_0, y_pred_0 = y_true[:, :, :, 0] * 255, y_pred[:, :, :, 0] * 255
    y_true_1, y_pred_1 = y_true[:, :, :, 1] * 0.05, y_pred[:, :, :, 1] * 0.05
    mse = tf.losses.mean_squared_error
    return mse(y_true_0, y_pred_0) + mse(y_true_1, y_pred_1)


def Asymmetry_Binary_Loss_2(y_true, y_pred):
    # 想要损失函数更加关心裂缝的标签值1
    y_true_0, y_pred_0 = y_true[:, :, :, 0], y_pred[:, :, :, 0]
    # y_true_0, y_pred_0 = y_true[:, :, :, 0] * 255, y_pred[:, :, :, 0] * 255
    y_true_1, y_pred_1 = y_true[:, :, :, 1], y_pred[:, :, :, 1]
    bcr = tf.losses.binary_crossentropy
    return bcr(y_true_0, y_pred_0) + bcr(y_true_1, y_pred_1)


def Constraints_Loss(y_true, y_pred):
    y_true = tf.ones_like(y_true[:, :, :, 0]) * 10
    y_pred_0 = y_pred[:, :, :, 0] * 10
    y_pred_1 = y_pred[:, :, :, 1] * 10
    y_pred = y_pred_0 + y_pred_1
    mse = tf.losses.mean_squared_error

    return mse(y_true, y_pred)


def dice_loss(y_true, y_pred, ep=1e-8):
    ep = tf.constant(ep, tf.float32)
    alpha = tf.constant(2, tf.float32)
    # y_true_0, y_pred_0 = tf.cast(y_true[:, :, :, 0], tf.float32), tf.cast(y_pred[:, :, :, 0].as_dtype(tf.float32),
    # tf.float32)
    y_true_0, y_pred_0 = y_true[:, :, :, 0], y_pred[:, :, :, 0]
    intersection = alpha * tf.cast(K.sum(y_pred_0 * y_true_0), tf.float32) + ep
    union = tf.cast(K.sum(y_pred_0), tf.float32) + tf.cast(K.sum(y_true_0), tf.float32) + ep
    loss = 1 - intersection / union

    return loss


def Total_loss(y_true, y_pred):

    return Asymmetry_Binary_Loss(y_true, y_pred) + Constraints_Loss(y_true, y_pred) + dice_loss(y_true, y_pred)


# KD损失函数-alpha=0.9
def S_KD_Loss(y_true, y_pred, alpha=0.9):

    soft_label_loss = Asymmetry_Binary_Loss(y_true, y_pred)

    return alpha * soft_label_loss * 10


def H_KD_Loss(y_true, y_pred, alpha=0.9):

    hard_label_loss = Asymmetry_Binary_Loss(y_true, y_pred)

    return (1 - alpha) * hard_label_loss * 10


def Precision(y_true, y_pred):
    """精确率"""
    tp = K.sum(K.round(K.clip(y_true[:, :, :, 0], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # predicted positives
    precision = tp / (pp + 1e-8)
    return precision


def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true[:, :, :, 0], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))   # true positives
    pp = K.sum(K.round(K.clip(y_true[:, :, :, 0], 0, 1)))  # possible positives
    recall = tp / (pp + 1e-8)
    return recall


def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def IOU(y_true: tf.Tensor,
        y_pred: tf.Tensor):
    predict = K.round(K.clip(y_pred[:, :, :, 0], 0, 1))
    Intersection = K.sum(K.round(K.clip(y_true[:, :, :, 0], 0, 1)) * predict)
    Union = K.sum(K.round(K.clip(y_true[:, :, :, 0], 0, 1)) + predict)
    iou = Intersection / (Union - Intersection + 1e-8)
    return iou


def iou_keras(y_true, y_pred):
    """
    Return the Intersection over Union (IoU).
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the IoU for the given label
    """
    label = 1
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(y_pred, label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)
