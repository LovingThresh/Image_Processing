import tensorflow as tf
import tensorflow.keras as keras
import os
import tf2lib as tl


# checkpoint

class CheckpointSaver(keras.callbacks.Callback):
    def __init__(self, checkpoints_directory):
        super(CheckpointSaver, self).__init__()
        self.checkpoints_directory = checkpoints_directory
        self.checkpoints_prefix = os.path.join(self.checkpoints_directory, "ckpt")

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoints = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)
        self.status = self.checkpoints.restore(tf.train.latest_checkpoint(self.checkpoints_directory))
        self.manager = tf.train.CheckpointManager(self.checkpoints, directory=self.checkpoints_prefix,
                                                  max_to_keep=3)
        self.manager.save()
