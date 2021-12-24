import tensorflow.keras as keras
import tensorflow as tf


def student_model(input_shape=(448, 448, 3),
                  output_channels=2,
                  dim=16,
                  n_downsamplings=2,
                  n_blocks=4,
                  Temperature=10,
                  Temperature_for_real=10
                  ):
    output_channels = output_channels + 1

    def _residual_block(c):
        x_dim = c.shape[-1]
        d = c

        # 为什么这里不用padding参数呢？使用到了‘REFLECT’
        d = keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = keras.layers.Conv2D(x_dim, 3, padding='valid', use_bias=False)(d)
        d = keras.layers.BatchNormalization()(d)
        d = keras.layers.ReLU()(d)

        d = keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = keras.layers.Conv2D(x_dim, 3, padding='valid', use_bias=False)(d)
        d = keras.layers.BatchNormalization()(d)

        return keras.layers.Add()([c, d])

    h = inputs = keras.Input(shape=input_shape)

    # 针对x进行膨胀卷积
    x = h
    x = keras.layers.ZeroPadding2D(padding=(3, 3))(x)
    x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='valid', dilation_rate=(3, 3), use_bias=False)(x)
    x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # 针对y进行长方形卷积
    y = h
    y1 = keras.layers.Conv2D(dim, (7, 3), strides=1, padding='same', use_bias=False)(y)
    y1 = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y1)
    y2 = keras.layers.Conv2D(dim, (3, 7), strides=1, padding='same', use_bias=False)(y)
    y2 = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(y2)
    y = keras.layers.Add()([y1, y2])
    y = keras.layers.Conv2D(dim, 3, padding='same', use_bias=False)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)

    # 针对h进行普通卷积
    h = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(h)
    h = keras.layers.Conv2D(dim, 3, padding='same', use_bias=False)(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.ReLU()(h)

    for _ in range(n_downsamplings):
        dim *= 2

        x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(6, 6), use_bias=False)(x)
        x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(6, 6), use_bias=False)(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        y1 = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y)
        y1 = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y1)
        y2 = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(y)
        y2 = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(y2)

        y = keras.layers.Add()([y1, y2])
        y = keras.layers.MaxPooling2D((2, 2))(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.ReLU()(y)

        h = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(h)
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = keras.layers.BatchNormalization()(h)
        h = keras.layers.ReLU()(h)

    for _ in range(n_blocks):
        h = _residual_block(h)
        x = _residual_block(x)
        y = _residual_block(y)

    for _ in range(n_downsamplings):
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = keras.layers.BatchNormalization()(h)
        h = keras.layers.ReLU()(h)

        # x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        # y = keras.layers.Dropout(0.5)(y)
        y = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.ReLU()(y)

    h = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(h)
    h = keras.layers.Conv2D(output_channels, 3, padding='same', use_bias=False)(h)

    x = keras.layers.Conv2D(output_channels, (3, 3), strides=(1, 1), padding='same', dilation_rate=(3, 3),
                            use_bias=False)(x)
    x = keras.layers.Conv2D(output_channels, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2),
                            use_bias=False)(x)

    y1 = keras.layers.Conv2D(output_channels, (7, 3), strides=1, padding='same', use_bias=False)(y)
    y1 = keras.layers.Conv2D(output_channels, (3, 1), strides=1, padding='same', use_bias=False)(y1)
    y2 = keras.layers.Conv2D(output_channels, (3, 7), strides=1, padding='same', use_bias=False)(y)
    y2 = keras.layers.Conv2D(output_channels, (1, 3), strides=1, padding='same', use_bias=False)(y2)
    y = keras.layers.Add()([y1, y2])
    y = keras.layers.Conv2D(output_channels, 3, padding='same', use_bias=False)(y)

    mix = keras.layers.Add()([h, x, y])

    attention_mask = tf.sigmoid(h[:, :, :, 0:1])
    content_mask = h[:, :, :, 1:]
    attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
    h = keras.layers.Multiply()([attention_mask, content_mask])

    attention_mask = tf.sigmoid(x[:, :, :, 0:1])
    content_mask = x[:, :, :, 1:]
    attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
    x = keras.layers.Multiply()([attention_mask, content_mask])

    attention_mask = tf.sigmoid(y[:, :, :, 0:1])
    content_mask = y[:, :, :, 1:]
    attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
    y = keras.layers.Multiply()([attention_mask, content_mask])

    attention_mask = tf.sigmoid(mix[:, :, :, 0:1])
    content_mask = mix[:, :, :, 1:]
    attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
    mix = keras.layers.Multiply()([attention_mask, content_mask])

    h = h / Temperature
    x = x / Temperature
    y = y / Temperature
    mix_for_real = mix / Temperature_for_real
    mix = mix / Temperature

    h = keras.layers.Softmax(name='Label_h')(h)
    x = keras.layers.Softmax(name='Label_x')(x)
    y = keras.layers.Softmax(name='Label_y')(y)
    mix = keras.layers.Softmax(name='Label_mix')(mix)
    mix_for_real = keras.layers.Softmax(name='Label_mix_for_real')(mix_for_real)

    return keras.Model(inputs=inputs, outputs=[h, x, y, mix, mix_for_real])


model = student_model()
model.summary()
