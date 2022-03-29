import numpy as np
from Metrics import *
from Layer import *
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================


# ==================================Res-Net======================================
def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ResnetGenerator(input_shape=(None, None, 3),
                    output_channels=2,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=8,
                    norm='instance_norm',
                    attention=False,
                    ShallowConnect=False):
    Norm = _get_norm_layer(norm)
    if attention:
        output_channels = output_channels + 1

    # 受保护的用法
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        # 为什么这里不用padding参数呢？使用到了‘REFLECT’
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = DilatedConv2D(k_size=3, rate=2, out_channel=dim, padding='SAME', name='dilatedConv')(h)
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)
    if ShallowConnect:
        f1 = h

    # 2
    for i in range(n_downsamplings):
        dim *= 2
        # x = h
        # x = keras.layers.Conv2D(dim, (7, 3), strides=1, padding='same', use_bias=False)(x)
        # x = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(x)
        # # x = keras.layers.Conv2D(dim, (3, 7), strides=1, padding='same', use_bias=False)(x)
        # # x = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(x)
        # x = keras.layers.MaxPooling2D((2, 2))(x)
        # x = Norm()(x)
        # x = tf.nn.relu(x)
        #
        # y = h
        # y = keras.layers.Conv2D(dim, (7, 3), strides=1, padding='same', use_bias=False)(y)
        # y = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y)
        # # x = keras.layers.Conv2D(dim, (3, 7), strides=1, padding='same', use_bias=False)(x)
        # # x = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(x)
        # y = keras.layers.MaxPooling2D((2, 2))(y)
        # y = Norm()(y)
        # y = tf.nn.relu(y)
        #
        # h = keras.layers.concatenate([x, y], -1)
        # h = keras.layers.Conv2D(dim, (1, 1), strides=1, padding='same', use_bias=False)(h)

        # y = h
        # y = DilatedConv2D(k_size=3, rate=3, out_channel=dim, padding='SAME', name='dilatedConv_y_0_{}'.format(i))(y)
        # y = DilatedConv2D(k_size=3, rate=2, out_channel=dim, padding='SAME', name='dilatedConv_y_1_{}'.format(i))(y)
        # y = keras.layers.MaxPooling2D((2, 2))(y)
        # y = Norm()(y)
        # y = tf.nn.relu(y)
        #
        #
        h = DilatedConv2D(k_size=3, rate=2, out_channel=dim, padding='SAME', name='dilatedConv{}'.format(i))(h)
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        if (i == 0) & ShallowConnect:
            f2 = h
        #
        # h = keras.layers.Add()([x, y, h])

    if ShallowConnect:
        f3 = h
    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    if ShallowConnect:
        h = keras.layers.concatenate([h, f3], axis=-1)

    # 4
    for _ in range(n_downsamplings):
        if (_ == 1) & ShallowConnect:
            h = keras.layers.concatenate([h, f2], axis=-1)
        dim //= 2
        # for _ in range(1):
        #     h = _residual_block(h)
        h = keras.layers.Dropout(0.5)(h)
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    if ShallowConnect:
        h = keras.layers.concatenate([h, f1], axis=-1)
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    # for _ in range(1):
    #     h = _residual_block(h)

    if input_shape == (227, 227, 3):
        h = keras.layers.Conv2D(output_channels, 8, padding='valid')(h)
    else:
        h = keras.layers.Conv2D(output_channels, 7, padding='valid', use_bias=False)(h)

    if attention:
        attention_mask = tf.sigmoid(h[:, :, :, 0])
        # attention_mask = tf.sigmoid(h[:, :, :, :1])
        content_mask = h[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        h = content_mask * attention_mask
        # content_mask.shape=(B,H,W,C[通道数是输入时的C，此例中为2])  attention_mask.shape=(B,H,W,1) *[可解释为expand] C)
    # h = tf.tanh(h)
    h = keras.layers.Softmax()(h)
    return keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


# ==================================U-Net=======================================

class_number = 2  # 分类数


def encoder(input_height, input_width):
    """
    语义分割的第一部分，特征提取，主要用到VGG网络， 函数式API

    :param input_height: 输入图像的长
    :param input_width: 输入图像的宽
    :return: 返回： 输入图像， 提取到的5个特征
    """

    # 输入
    img_input = Input(shape=(input_height, input_width, 3))

    # 三行为一个结构单元，size减半
    # 227,227,3 -> 224, 224, 64
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((4, 4), strides=(1, 1))(x)
    f1 = x  # 暂存提取的特征

    # 224,224,64 -> 112, 112,128
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f2 = x

    # 112, 112, 128 -> 56, 56, 256
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f3 = x

    # 56, 56, 256 -> 28, 28, 512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f4 = x

    # 28, 28, 512 -> 14, 14, 512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]


def decoder(feature_map_list, class_number, input_height=227, input_width=227, encoder_level=3):
    """
    语义分割的后半部分，上采样，将图片放大

    :param feature_map_list: 特征图（多个），由encoder得到
    :param class_number: 分类数
    :param input_height: 输入图像长
    :param input_width: 输入图像宽
    :param encoder_level: 利用的特征图，这里利用f4
    :return: output ， 返回放大的特征图 （224, 224, 2）
    """

    # 获取一个特征图， 特征图来源encoder里面的f1, f2, f3, f4, f5;这里获取到f4
    feature_map = feature_map_list[encoder_level]

    # 解码过程， 以下（28, 28, 512） -> (224, 224, 2)

    # f4.shape=(28, 28, 512) -> (28, 28, 512)
    x = ZeroPadding2D((1, 1))(feature_map)
    x = Conv2D(512, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样， 图像长宽扩大2倍， (28, 28, 512) -> (56, 56, 256)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样， 图像长宽扩大2倍， (56, 56, 256) -> (112, 112, 128)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样， 图像长宽扩大2倍， (112, 112, 128) -> (224, 224, 64)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 再进行一次卷积， 将通道数变为2（要分类的数目） (224, 224, 64) -> (224, 224, 2)
    x = Conv2D(class_number, (3, 3), padding='same')(x)
    # reshape：(224, 224, 2) -> (224*224, 2)

    x = Conv2DTranspose(class_number, (5, 5), (1, 1), 'valid')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(class_number, (5, 5), (1, 1), 'valid')(x)
    x = BatchNormalization()(x)

    x = keras.layers.Softmax()(x)

    return x


def U_Net(Height=227, Width=227):
    """
    model 的主程序， 语义分割， 分两部分， 第一部分特征提取， 第二部分放大图片

    :param Height: 图像高
    :param Width:  图像宽
    :return: model
    """

    # 第一部分 编码， 提取特征， 图像size减小， 通道增加
    img_input, feature_map_list = encoder(input_height=Height, input_width=Width)

    # 第二部分 解码， 将图像上采样， size放大， 通道减小
    output = decoder(feature_map_list, class_number=class_number, input_height=Height, input_width=Width,
                     encoder_level=3)

    # 构建模型
    model = Model(img_input, output)

    # model.summary()

    return model


def TeacherNet():
    conv_model = keras.models.load_model(r'I:\Image Processing\output\2021-09-17-13-40-41.901808\checkpoint\ep390'
                                         r'-val_loss0.004-val_acc0.999.h5',
                                         custom_objects={'Precision': A_Precision,
                                                         'Recall': A_Recall,
                                                         'F1': A_F1,
                                                         'IOU': A_IOU,
                                                         'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss
                                                         })
    conv_model.trainable = False
    # 组建网络
    model = tf.keras.models.Sequential()
    model.add(conv_model)
    model.add(tf.keras.layers.Softmax(axis=3))

    return model


# Teacher_Label已经构建好了，接下来进行StudentsNet的构建
# 首先按照原始模型缩小进行实验
# 两种形态进行对比
def StudentNet(input_shape=(512, 512, 3),
               output_channels=2,
               dim=32,
               n_downsamplings=2,
               n_blocks=4,
               norm='instance_norm',
               attention=False,
               Separable_convolution=False):
    Norm = _get_norm_layer(norm)
    if attention:
        output_channels = output_channels + 1
    if Separable_convolution:
        layer_Conv2D = keras.layers.SeparableConv2D
    else:
        layer_Conv2D = keras.layers.Conv2D

    # 受保护的用法
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        # 为什么这里不用padding参数呢？使用到了‘REFLECT’
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]])

        h = layer_Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]])
        h = layer_Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]])
    h = layer_Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = layer_Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]])
    if input_shape == (227, 227, 3):
        h = layer_Conv2D(output_channels, 8, padding='valid')(h)
    elif input_shape == (512, 512, 3):
        h = layer_Conv2D(output_channels, 7, padding='valid')(h)
    if attention:
        attention_mask = tf.sigmoid(h[:, :, :, 0])
        # attention_mask = tf.sigmoid(h[:, :, :, :1])
        content_mask = h[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        h = content_mask * attention_mask
        # content_mask.shape=(B,H,W,C[通道数是输入时的C，此例中为2])  attention_mask.shape=(B,H,W,1) *[可解释为expand] C)
    h = keras.layers.Activation('tanh', name='Output_Label')(h)
    soft_target = keras.layers.Softmax(axis=3, name='Soft_Label')(h)

    return keras.Model(inputs=inputs, outputs=[h, soft_target])
    # return keras.Model(inputs=inputs, outputs=h)


# =============================Attention Cycle GAN==============================
def AttentionCycleGAN_v1_Generator(input_shape=(227, 227, 3), output_channel=3,
                                   n_downsampling=2, n_ResBlock=9,
                                   norm='batch_norm', attention=False):
    Norm = _get_norm_layer(norm)
    input_layer = keras.Input(shape=input_shape)
    h = tf.pad(input_layer, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    model_layer_1 = keras.models.Sequential([
        Conv2D(64, (7, 7), (1, 1), 'valid'),
        Norm(),
    ], name='input_layer_1')
    h = model_layer_1(h)

    n_downsampling = n_downsampling
    n_ResBlock = n_ResBlock
    if attention:
        output_channel = output_channel + 1
    for i in range(n_downsampling):
        mult = 2 ** i
        model_layer_2 = keras.models.Sequential([
            Conv2D(64 * mult * 2, (3, 3), (2, 2), 'same'),
            Norm(),
            ReLU(),
        ])
        h = model_layer_2(h)

    mult = 2 ** n_downsampling
    for i in range(n_ResBlock):
        x = h

        model_layer_3 = keras.models.Sequential([
            Conv2D(64 * mult, (3, 3), padding='valid'),
            Norm(),
        ])

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = model_layer_3(h)
        h = ReLU()(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = model_layer_3(h)

        h = x + h

    upsampling = n_downsampling

    for i in range(upsampling):
        model_layer_4 = keras.models.Sequential([
            Conv2DTranspose(64 * mult / 2, (3, 3), (2, 2), 'same'),
            Norm(),
            ReLU(),
        ])
        h = model_layer_4(h)
        mult = mult / 2

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = Conv2D(output_channel, (8, 8), (1, 1), 'valid')(h)
    h = Norm()(h)
    h = tf.tanh(h)
    result_layer = h
    if attention:
        attention_mask = tf.sigmoid(h[:, :, :, :1])
        content_mask = h[:, :, :, 1:]
        attention_mask = tf.concat([attention_mask, attention_mask, attention_mask], axis=3)
        result_layer = content_mask * attention_mask + input_layer * (1 - attention_mask)

    return keras.Model(inputs=input_layer, outputs=result_layer)


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def get_config(self):
        pass

    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (
                    1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate


# 模型的子类写法，这是一个简单的实例，不用于使用
class MyModel(tf.keras.Model):

    def get_config(self):
        pass

    # 如果不写get_config,将无法在TensorBoard中载入模型图(model Graph)

    def __init__(self, num_classes=10, input_shape=None):
        super(MyModel, self).__init__(name='my_model')
        self.data_input_shape = input_shape
        self.num_classes = num_classes
        # 定义自己需要的层
        self.dense_1 = Dense(32, activation='relu', input_shape=(100, 32))  #
        self.dense_2 = Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        # 定义前向传播
        # 使用在 (in `__init__`)定义的层
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

    def model(self):
        x = Input(shape=self.data_input_shape)
        return Model(inputs=[x], outputs=self.call(x))


# ==============================================================================
# =                          Attention Module Layer                            =
# ==============================================================================
class SELayer(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.avg_poll = GlobalAvgPool2D()
        self.sigmoid = tf.keras.activations.sigmoid

    def fc(self, x):
        x = Dense(self.channels // self.reduction, use_bias=False)(x),
        x = ReLU()(x),
        x = Dense(self.channels, use_bias=False)(x)

        return x

    def call(self, inputs, *args, **kwargs):
        b, _, _, c = inputs.shape
        y = self.avg_poll(inputs)
        y = tf.reshape(y, (y.shape[0], 1, 1, y.shape[1]))
        y = self.fc(y)
        y = self.sigmoid(y)
        # mask.shape = (B, 1, 1, C)
        return inputs * y, y


# 何叶师姐的注意力机制，这是好像是空间域的提取
class AttentionModuleHeye(keras.layers.Layer):
    def __init__(self, F_g, F_l, F_int):
        self.F_g = F_g
        self.F_l = F_l
        self.F_int = F_int
        super(AttentionModuleHeye, self).__init__()
        self.sigmoid = keras.activations.sigmoid
        self.relu = ReLU()

    def W_g(self, x):
        x = Conv2D(self.F_int, (1, 1), (1, 1), padding='same', use_bias=True)(x),
        x = BatchNormalization()(x)
        return x

    def W_x(self, x):
        x = Conv2D(self.F_int, (1, 1), (1, 1), padding='same', use_bias=True)(x),
        x = BatchNormalization()(x)
        return x

    def psi(self, x):
        x = Conv2D(1, (1, 1), (1, 1), padding='same', use_bias=True)(x),
        x = BatchNormalization()(x)
        return x

    def call(self, inputs, g=None):
        g1 = self.W_g(g)
        x1 = self.W_x(inputs)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.sigmoid(psi)
        return inputs * psi


# ==============================================================================
# =                               Network Layer                                =
# ==============================================================================

# ==============================PixelShuffle====================================


def squeeze_middle2axes_operator(x4d, C, output_size):
    keras.backend.set_image_data_format('channels_first')
    shape = tf.shape(x4d)  # get dynamic tensor shape
    x4d = tf.reshape(x4d, [shape[0], shape[1], shape[2] * 2, shape[4] * 2])
    return x4d


def squeeze_middle2axes_shape(output_size):
    in_batch, C, in_rows, _, in_cols, _ = output_size

    if None in [in_rows, in_cols]:
        output_shape = (in_batch, C, None, None)
    else:
        output_shape = (in_batch, C, in_rows, in_cols)
    return output_shape


class pixelshuffle(tf.keras.layers.Layer):
    """Sub-pixel convolution layer.
    See https://arxiv.org/abs/1609.05158
    """

    def __init__(self, scale, trainable=False, **kwargs):
        self.scale = scale
        super().__init__(trainable=trainable, **kwargs)

    def call(self, t, *args, **kwargs):
        upscale_factor = self.scale
        input_size = t.shape.as_list()
        dimensionality = len(input_size) - 2
        new_shape = self.compute_output_shape(input_size)
        C = new_shape[1]

        output_size = new_shape[2:]
        x = [upscale_factor] * dimensionality
        old_h = input_size[-2] if input_size[-2] is not None else -1
        old_w = input_size[-1] if input_size[-1] is not None else -1

        shape = tf.shape(t)
        t = tf.reshape(t, [-1, C, x[0], x[1], shape[-2], shape[-1]])

        perms = [0, 1, 5, 2, 4, 3]
        t = tf.transpose(t, perm=perms)
        t = Lambda(squeeze_middle2axes_operator, output_shape=squeeze_middle2axes_shape,
                   arguments={'C': C, 'output_size': output_size})(t)
        t = tf.transpose(t, [0, 1, 3, 2])
        return t

    def compute_output_shape(self, input_shape):
        r = self.scale
        rrC, H, W = np.array(input_shape[1:])
        assert rrC % (r ** 2) == 0
        height = H * r if H is not None else -1
        width = W * r if W is not None else -1

        return input_shape[0], rrC // (r ** 2), height, width


def ResnetGenerator_with_ThreeChannel(input_shape=(448, 448, 3),
                                      output_channels=2,
                                      dim=64,
                                      n_downsamplings=2,
                                      n_blocks=8,
                                      norm='instance_norm',
                                      attention=False,
                                      ShallowConnect=False,
                                      Temperature=0,
                                      StudentNet=False):
    Norm = _get_norm_layer(norm)
    if attention:
        output_channels = output_channels + 1

    # 受保护的用法
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        # 为什么这里不用padding参数呢？使用到了‘REFLECT’
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')

        h = keras.layers.Conv2D(dim, (3, 3), 1, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h = keras.layers.Conv2D(dim, (3, 3), 1, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape, name='input_0')
    # h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')

    # 针对x进行膨胀卷积
    x = h
    # x = DilatedConv2D(k_size=3, rate=3, out_channel=dim, padding='VALID', name='dilatedConv0_1')(x)
    x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')
    x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='valid', dilation_rate=(3, 3), use_bias=False)(x)
    # x = DilatedConv2D(k_size=3, rate=2, out_channel=dim, padding='SAME', name='dilatedConv0_2')(x)
    x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(x)
    x = Norm()(x)
    x = tf.nn.relu(x)

    # 针对y进行长方卷积
    y = h
    # y = tf.pad(y, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')
    y1 = keras.layers.Conv2D(dim, (7, 3), strides=1, padding='same', use_bias=False)(y)
    y1 = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y1)
    y2 = keras.layers.Conv2D(dim, (3, 7), strides=1, padding='same', use_bias=False)(y)
    y2 = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(y2)
    y = keras.layers.Add()([y1, y2])
    y = keras.layers.Conv2D(dim, 7, padding='same', use_bias=False)(y)
    # y = Norm()(y)
    # y = tf.nn.relu(y)
    # 1

    # h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')
    h = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(h)
    h = keras.layers.Conv2D(dim, 7, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)
    if ShallowConnect:
        f1 = h

    # 2
    for i in range(n_downsamplings):
        dim *= 2

        x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(3, 3), use_bias=False)(x)
        x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = Norm()(x)
        x = tf.nn.relu(x)

        y1 = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y)
        y1 = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y1)
        y2 = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(y)
        y2 = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(y2)

        y = keras.layers.Add()([y1, y2])
        y = keras.layers.MaxPooling2D((2, 2))(y)
        y = Norm()(y)
        y = tf.nn.relu(y)

        h = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(h)
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        if (i == 0) & ShallowConnect:
            f2 = h

        # h = keras.layers.Add()([x, y, h])

    if ShallowConnect:
        f3 = h
    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)
        x = _residual_block(x)
        y = _residual_block(y)

    if ShallowConnect:
        h = keras.layers.concatenate([h, f3], axis=-1)

    # 4
    for _ in range(n_downsamplings):
        if (_ == 1) & ShallowConnect:
            h = keras.layers.concatenate([h, f2], axis=-1)
        dim //= 2
        # for _ in range(1):
        #     h = _residual_block(h)
        # h = keras.layers.Dropout(0.5)(h)
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        # x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        # y = keras.layers.Dropout(0.5)(y)
        y = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(y)
        y = Norm()(y)
        y = tf.nn.relu(y)

    # 5
    if ShallowConnect:
        h = keras.layers.concatenate([h, f1], axis=-1)
    # h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')
    # for _ in range(1):
    #     h = _residual_block(h)
    x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')
    # y = tf.pad(y, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')

    if input_shape == (227, 227, 3):
        h = keras.layers.Conv2D(output_channels, 8, padding='valid')(h)
    else:
        h = keras.layers.Conv2D(output_channels, 7, padding='same', use_bias=False)(h)

        x = keras.layers.Conv2D(output_channels, (3, 3), strides=(1, 1), padding='valid', dilation_rate=(3, 3),
                                use_bias=False)(x)
        x = keras.layers.Conv2D(output_channels, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2),
                                use_bias=False)(x)

        y1 = keras.layers.Conv2D(output_channels, (7, 3), strides=1, padding='same', use_bias=False)(y)
        y1 = keras.layers.Conv2D(output_channels, (3, 1), strides=1, padding='same', use_bias=False)(y1)
        y2 = keras.layers.Conv2D(output_channels, (3, 7), strides=1, padding='same', use_bias=False)(y)
        y2 = keras.layers.Conv2D(output_channels, (1, 3), strides=1, padding='same', use_bias=False)(y2)
        y = keras.layers.Add()([y1, y2])
        y = keras.layers.Conv2D(output_channels, 7, padding='same', use_bias=False)(y)

        mix = keras.layers.Add()([h, y, x])
    if attention:
        attention_mask = tf.sigmoid(h[:, :, :, 0])
        # attention_mask = tf.sigmoid(h[:, :, :, :1])
        content_mask = h[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        h = content_mask * attention_mask

        attention_mask = tf.sigmoid(x[:, :, :, 0])
        # attention_mask = tf.sigmoid(h[:, :, :, :1])
        content_mask = x[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        x = content_mask * attention_mask

        attention_mask = tf.sigmoid(y[:, :, :, 0])
        # attention_mask = tf.sigmoid(h[:, :, :, :1])
        content_mask = y[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        y = content_mask * attention_mask
        # content_mask.shape=(B,H,W,C[通道数是输入时的C，此例中为2])  attention_mask.shape=(B,H,W,1) *[可解释为expand] C)

        attention_mask = tf.sigmoid(mix[:, :, :, 0])
        # attention_mask = tf.sigmoid(h[:, :, :, :1])
        content_mask = mix[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        mix = content_mask * attention_mask
    # h = tf.tanh(h)

    if (Temperature != 0) & StudentNet:
        h = h / Temperature
        x = x / Temperature
        y = y / Temperature
        mix_for_real = mix
        mix = mix / Temperature
        mix_for_real_low_T = keras.layers.Softmax(name='Label_mix_for_real_low_T')(mix_for_real)
        mix_for_real = keras.layers.Softmax(name='Label_mix_for_real')(mix_for_real / Temperature)

    h = keras.layers.Softmax(name='Label_h')(h)
    x = keras.layers.Softmax(name='Label_x')(x)
    y = keras.layers.Softmax(name='Label_y')(y)
    mix = keras.layers.Softmax(name='Label_mix')(mix)

    return keras.Model(inputs=inputs, outputs=[h, x, y, mix, mix_for_real])
