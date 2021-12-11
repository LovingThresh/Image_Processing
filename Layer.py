import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import Variable
from tensorflow.python.ops.init_ops_v2 import RandomNormal


class DilatedConv2D(keras.layers.Layer):
    w_init: RandomNormal
    weight: Variable

    def __init__(self, k_size=3, rate=1, out_channel=32, padding='SAME', name='dilatedConv2D'):
        super(DilatedConv2D, self).__init__()
        self.k_size = k_size
        self.rate = rate
        self.uints = out_channel
        self.padding = padding
        self._name = name

    def build(self, input_shape):
        self.w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(
            initial_value=self.w_init(shape=(self.k_size, self.k_size, input_shape[-1], self.uints),
                                      dtype='float32'), trainable=True)

    def call(self, inputs,  **kwargs):  # Defines the computation from inputs to outputs

        return tf.nn.atrous_conv2d(value=inputs, filters=self.weight, rate=self.rate, padding=self.padding,
                                   name=self._name)


