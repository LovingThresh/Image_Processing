# 这个文件创建的目的是为了复现(复制)目前比较主流的分割算法
# 以作为参照，真实地了解目前架构的性能
# import tensorflow as tf
import tensorflow.keras as keras

pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                 "releases/download/v0.1/" \
                 "vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
# model是网络对象
model = object


# ==========================================================================================
#                                Encoder
# ==========================================================================================

# 首先是最基础的VGG16_Encoder，根据依靠的架构也各有不同
def VGG_16_Encoder(input_shape=(224, 224, 3), pretrained=False):
    global model
    assert input_shape[0] % 32 == 0
    assert input_shape[1] % 32 == 0

    input_layer = keras.layers.Input(shape=input_shape)

    # block1
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_layer)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_maxpool1')(x)
    f1 = x

    # block2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_maxpool1')(x)
    f2 = x

    # block3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_maxpool1')(x)
    f3 = x

    # block4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_maxpool1')(x)
    f4 = x

    # block5
    # 经过block5之后，模型的输出为W/32， H/32， 512
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_maxpool1')(x)
    f5 = x

    features = [f1, f2, f3, f4, f5]
    if pretrained == 'imagenet':
        VGG_Weights_path = keras.utils.get_file(pretrained_url.split("/")[-1], pretrained_url)
        model = keras.Model(inputs=input_layer, outputs=x) \
            .load_weights(VGG_Weights_path, by_name=True, skip_mismatch=True)
    if not pretrained:
        model = keras.Model(input_layer, x)

    return model, features


# ==========================================================================================
#                                Decoder
# ==========================================================================================

# 首先是分类器的Decoder，其对应的就是VGG16网络
def ClassificationDecoder(Encoder, num_class: int):
    global model
    Encoder_out_layer = Encoder.output
    x = keras.layers.Dense(4096, activation='relu')(Encoder_out_layer)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(num_class, activation='relu')(x)

    x = keras.layers.Softmax()(x)

    model = keras.Model(inputs=Encoder.input, outputs=x)

    return model


# SegmentationDecoder 对应的是U-Net的写法
def SegmentationDecoder(Encoder, feature_list=None, out_num=None):
    global model
    f1, f2, f3, f4, _ = feature_list
    Encoder_out_layer = Encoder.output

    # transpose_block1
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block1_conv1')(Encoder_out_layer)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block1_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block1_conv3')(x)
    x = keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block1_Tconv1')(x)

    # transpose_block2
    x = keras.layers.concatenate([x, f4], axis=-1)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block2_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block2_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block2_conv3')(x)
    x = keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block2_Tconv1')(x)

    # transpose_block3
    x = keras.layers.concatenate([x, f3], axis=-1)
    x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block3_conv3')(x)
    x = keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block3_Tconv1')(x)

    # transpose_block4
    x = keras.layers.concatenate([x, f2], axis=-1)
    x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block4_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block4_conv2')(x)
    x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block4_conv3')(x)
    x = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block4_Tconv1')(x)

    # transpose_block5
    x = keras.layers.concatenate([x, f1], axis=-1)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block5_conv1')(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block5_conv2')(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block5_conv3')(x)
    x = keras.layers.Conv2DTranspose(out_num, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block5_Tconv1')(x)

    model = keras.Model(inputs=Encoder.input, outputs=x, name='U-Net')

    return model


# FCN系列网络-FCN-8s，FCN-16s，FCN-32s
# 在经典的VGGNet的基础上，把VGG网络最后的全连接层全部去掉，换为卷积层
def FCN_32sEncoder(Encoder, feature_list=None, out_num=None):
    global model
    f1, f2, f3, f4, _ = feature_list
    Encoder_out_layer = Encoder.output

    # D_block1
    x = keras.layers.Conv2D(4096, (7, 7), strides=(1, 1), activation='relu', padding='same',
                            name='D_block1_conv1')(Encoder_out_layer)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(4096, (1, 1), strides=(1, 1), padding='same', activation='relu',
                            name='D_block1_conv2')(x)
    x = keras.layers.Dropout(0.5)(x)

    # D_block2
    x = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                            name='Seg_feats')(x)

    # D_block3
    x = keras.layers.Conv2DTranspose(out_num, (64, 64), strides=(32, 32), use_bias=False, padding='same',
                                     name='D_block3_Tconv1')(x)

    model = keras.models.Model(inputs=Encoder.input, outputs=x, name='FCN_32s')

    return model


def FCN_8sEncoder(Encoder, feature_list=None, out_num=None):
    global model
    f1, f2, f3, f4, _ = feature_list
    Encoder_out_layer = Encoder.output

    # D_block1
    x = keras.layers.Conv2D(4096, (7, 7), strides=(1, 1), activation='relu', padding='same',
                            name='D_block1_conv1')(Encoder_out_layer)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(4096, (1, 1), strides=(1, 1), padding='same', activation='relu',
                            name='D_block1_conv2')(x)
    x = keras.layers.Dropout(0.5)(x)

    # D_block2
    x = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                            name='Seg_feats')(x)

    # D_block3
    x = keras.layers.Conv2DTranspose(out_num, (4, 4), (2, 2), use_bias=False,
                                     name='D_block3_Tconv1')(x)

    # D_block4
    f4 = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                             name='D_block4_conv1')(f4)
    x = keras.layers.Add(name='D_block4_add1')([x, f4])
    x = keras.layers.Conv2DTranspose(out_num, (4, 4), strides=(2, 2), use_bias=False,
                                     name='D_block4_Tconv1')(x)

    # D_block5
    f3 = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                             name='D_block5_conv1')(f3)
    x = keras.layers.Add(name='D_block5_add1')([x, f3])
    x = keras.layers.Conv2DTranspose(out_num, (16, 16), strides=(8, 8), use_bias=False,
                                     name='D_block5')(x)

    model = keras.models.Model(inputs=Encoder.input, outputs=x, name='FCN_8s')

    return model

# DeepLab系列分割网络
# def DeepLabV1(Encoder, feature_list=None, out_num=None):
