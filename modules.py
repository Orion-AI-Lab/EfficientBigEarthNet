import tensorflow as tf
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    AveragePooling2D,
    Reshape,
    Dense,
    multiply,
    Concatenate,
    Conv2D,
    Conv1D,
    Add,
    Activation,
    Lambda,
    BatchNormalization,
    Activation,
    DepthwiseConv2D,
)
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid, relu, swish
import math


def eca_module(inputs_tensor=None, num=None, gamma=2, b=1):
    channels = inputs_tensor.shape[-1]
    t = int(abs((math.log(channels, 2) + b) / gamma))
    k = t if t % 2 else t + 1
    x_global_avg_pool = GlobalAveragePooling2D()(inputs_tensor)
    x = Reshape((channels, 1))(x_global_avg_pool)
    x = Conv1D(1, kernel_size=k, padding="same")(x)
    x = Activation(sigmoid)(x)
    x = Reshape((1, 1, channels))(x)
    output = multiply([inputs_tensor, x])
    return output


def ghost_module(x, out_channels, kernel_size, strides):
    x1 = Conv2D(
        out_channels // 2,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
    )(x)
    x2 = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(x1)
    x2 = Activation(relu)(x2)
    x2 = DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
    )(x2)
    x2 = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(x1)
    x2 = Activation(relu)(x2)
    return Concatenate()([x1, x2])


def coord_module(x, reduction=32):
    def coord_act(x):
        tmpx = tf.nn.relu6(x + 3) / 6
        x = x * tmpx
        return x

    x_shape = x.get_shape().as_list()
    [b, h, w, c] = x_shape
    x_h = AveragePooling2D(pool_size=(1, w), strides=1)(x)
    x_w = AveragePooling2D(pool_size=(h, 1), strides=1)(x)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])

    y = tf.concat([x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = Conv2D(mip, (1, 1), strides=1, padding="VALID")(y)
    y = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(y)
    y = coord_act(y)

    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    a_h = Conv2D(c, (1, 1), strides=1, padding="VALID", activation=sigmoid)(x_h)
    a_w = Conv2D(c, (1, 1), strides=1, padding="VALID", activation=sigmoid)(x_w)

    out = x * a_h * a_w
    return out


def se_module(input_feature, ratio=16):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel = input_feature.shape[-1]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    se_feature = Dense(
        channel // ratio,
        activation="swish",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )(se_feature)
    se_feature = Dense(
        channel,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def cbam_module(cbam_feature, ratio=16):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):

    channel = input_feature.shape[-1]
    shared_layer_one = Dense(
        channel // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_layer_two = Dense(
        channel, kernel_initializer="he_normal", use_bias=True, bias_initializer="zeros"
    )

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation("sigmoid")(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    cbam_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(concat)

    return multiply([input_feature, cbam_feature])
