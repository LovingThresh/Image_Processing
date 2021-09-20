import tensorflow as tf
import tensorflow.keras as keras
import os
import tf2lib as tl


# checkpoint
# 目前存在的问题是只能输出最后一个checkpoints
# 2021/9/20 问题已经解决，目前可以输出指定的checkpoint了
class CheckpointSaver(keras.callbacks.Callback):
    def __init__(self, manager):
        super(CheckpointSaver, self).__init__()
        self.manager = manager

    def on_epoch_end(self, epoch, logs=None):
        self.manager.save()
