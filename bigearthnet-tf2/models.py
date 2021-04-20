# -*- coding: utf-8 -*-
#
# The models
#

import numpy as np
import tensorflow as tf
import math
import six
from tensorflow.keras.layers import Input, Add, Dense, Flatten, Activation, Lambda, Conv2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, VGG16, VGG19
from einops.layers.tensorflow import Rearrange
from tensorflow.python.ops import nn
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from inputs import BAND_STATS

SEED = 42

MODELS_CLASS = {
    "dense": "BigEarthModel",
    "ResNet50": "ResNet50BigEarthModel",
    "ResNet101": "ResNet101BigEarthModel",
    "ResNet152": "ResNet152BigEarthModel",
    "VGG16": "VGG16BigEarthModel",
    "VGG19": "VGG19BigEarthModel",
    "KBranch": "DNN_model",
    "DenseNet121": "DenseNet121BigEarthModel",
    "DenseNet161": "DenseNet169BigEarthModel",
    "DenseNet201": "DenseNet201BigEarthModel",
    "ResNet": "ResNet",
    "EfficientNet": "EfficientNet",
    "EfficientNetB0": "EfficientNetB0",
    "EfficientNetB1": "EfficientNetB1",
    "EfficientNetB2": "EfficientNetB2",
    "EfficientNetB3": "EfficientNetB3",
    "EfficientNetB4": "EfficientNetB4",
    "EfficientNetB5": "EfficientNetB5",
    "EfficientNetB6": "EfficientNetB6",
    "EfficientNetB7": "EfficientNetB7",
    "ViT":"ViT"
}


class BigEarthModel:
    def __init__(self, nb_class, dtype="float64"):
        self._nb_class = nb_class
        self._dtype = dtype

        self._inputB04 = Input(shape=(120, 120,), dtype=tf.float32)
        self._inputB03 = Input(shape=(120, 120,), dtype=tf.float32)
        self._inputB02 = Input(shape=(120, 120,), dtype=tf.float32)
        self._inputB08 = Input(shape=(120, 120,), dtype=tf.float32)
        bands_10m = tf.keras.backend.stack(
            [self._inputB04, self._inputB03, self._inputB02, self._inputB08], axis=3
        )
        print("10m shape: {}".format(bands_10m.shape))

        self._inputB05 = Input(shape=(60, 60,), dtype=tf.float32)
        self._inputB06 = Input(shape=(60, 60,), dtype=tf.float32)
        self._inputB07 = Input(shape=(60, 60,), dtype=tf.float32)
        self._inputB8A = Input(shape=(60, 60,), dtype=tf.float32)
        self._inputB11 = Input(shape=(60, 60,), dtype=tf.float32)
        self._inputB12 = Input(shape=(60, 60,), dtype=tf.float32)
        bands_20m = tf.stack(
            [
                self._inputB05,
                self._inputB06,
                self._inputB07,
                self._inputB8A,
                self._inputB11,
                self._inputB12,
            ],
            axis=3,
        )
        bands_20m = tf.image.resize(
            bands_20m, [120, 120], method=tf.image.ResizeMethod.BICUBIC
        )
        print("20m shape: {}".format(bands_20m.shape))

        # TODO: Should we include those? The bigearthnet authors do not.
        self._inputB01 = Input(shape=(20, 20,), dtype=tf.float32)
        self._inputB09 = Input(shape=(20, 20,), dtype=tf.float32)
        bands_60m = tf.keras.backend.stack([self._inputB01, self._inputB09], axis=3)
        bands_60m = tf.image.resize(
            bands_60m, [120, 120], method=tf.image.ResizeMethod.BICUBIC
        )
        print("60m shape: {}".format(bands_60m.shape))

        allbands = tf.concat([bands_10m, bands_20m, bands_60m], axis=3)
        # allbands = tf.concat([bands_10m, bands_20m], axis=3)
        self.band_dict = {'band_10': bands_10m, 'band_20': bands_20m, 'band_60': bands_60m}
        self.feature_size = 128
        self.nb_bands_10m = 4
        self.nb_bands_20m = 6
        self.nb_bands_60m = 2
        self.bands_10m = self.band_dict['band_10']
        self.bands_20m = self.band_dict['band_20']
        self.bands_60m = self.band_dict['band_60']
        self._num_bands = allbands.shape[3]
        print("allbands shape: {}".format(allbands.shape))

        inputs = [
            self._inputB01,
            self._inputB02,
            self._inputB03,
            self._inputB04,
            self._inputB05,
            self._inputB06,
            self._inputB07,
            self._inputB08,
            self._inputB8A,
            self._inputB09,
            self._inputB11,
            self._inputB12,
        ]

        # create internal model
        self._logits = self._create_model_logits(allbands)

        # Add one last dense layer with biases and sigmoid activation
        # This is like having nb_class separate binary classifiers
        self._output = Dense(
            units=self._nb_class, dtype=self._dtype, activation="sigmoid", use_bias=True
        )(self._logits)

        self._model = Model(inputs=inputs, outputs=self._output)
        self._logits_model = Model(inputs=inputs, outputs=self._logits)

    @property
    def model(self):
        return self._model

    @property
    def logits_model(self):
        return self._logits_model

    def _create_model_logits(self, allbands):
        x = Flatten(dtype=self._dtype)(allbands)
        x = Dense(128, dtype=self._dtype)(x)
        x = Dense(64, dtype=self._dtype)(x)
        return x


class DenseNet121BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = tf.keras.applications.DenseNet121(
            include_top=False, weights=None, input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class DenseNet169BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = tf.keras.applications.DenseNet169(
            include_top=False, weights=None, input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class DenseNet201BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = tf.keras.applications.DenseNet201(
            include_top=False, weights=None, input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class ResNet50BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = ResNet50(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling='avg',
        )(allbands)

        return x


class ResNet101BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = ResNet101(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling='avg',
        )(allbands)

        return x


class ResNet152BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = ResNet152(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling='avg',
        )(allbands)

        return x


class VGG16BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        x = VGG16(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling='avg',
        )(allbands)

        return x


class VGG19BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = VGG19(
            include_top=False,
            weights=None,
            input_shape=(120, 120, num_bands),
            pooling='avg',
        )(allbands)

        return x


class EfficientNet(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        self.width_coefficient = 1.33
        self.depth_coefficient = 1.125
        self.dropout_rate = 0.2

        def swish(x):
            return x * tf.nn.sigmoid(x)

        def round_filters(filters, multiplier):
            depth_divisor = 8
            min_depth = None
            min_depth = min_depth or depth_divisor
            filters = filters * multiplier
            new_filters = max(min_depth, int(
                filters + depth_divisor / 2) // depth_divisor * depth_divisor)
            if new_filters < 0.9 * filters:
                new_filters += depth_divisor
            return int(new_filters)

        def round_repeats(repeats, multiplier):
            if not multiplier:
                return repeats
            return int(math.ceil(multiplier * repeats))

        def conv_bn(x, filters, kernel_size, strides, activation=True):
            x = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding='SAME')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            if activation:
                x = swish(x)
            return x

        def depthwiseConv_bn(x, kernel_size, strides):
            x = tf.keras.layers.DepthwiseConv2D(kernel_size,
                                                padding='same',
                                                strides=strides)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(tf.nn.relu6)(x)
            return x

        def Squeeze_excitation_layer(x):
            inputs = x
            squeeze = inputs.shape[-1] / 2
            excitation = inputs.shape[-1]
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(squeeze)(x)
            x = swish(x)
            x = tf.keras.layers.Dense(excitation)(x)
            x = tf.keras.layers.Activation('sigmoid')(x)
            x = tf.keras.layers.Reshape((1, 1, excitation))(x)
            x = inputs * x
            return x

        def MBConv_idskip(x, filters, drop_connect_rate, kernel_size, strides, t):
            x_input = x
            x = conv_bn(x, filters=x.shape[-1] * t, kernel_size=1, strides=1)
            x = depthwiseConv_bn(x, kernel_size=kernel_size, strides=strides)
            x = Squeeze_excitation_layer(x)
            x = swish(x)
            x = conv_bn(x, filters=filters, kernel_size=1, strides=1, activation=False)

            if strides == 1 and x.shape[-1] == x_input.shape[-1]:
                if drop_connect_rate:
                    x = tf.keras.layers.Dropout(rate=drop_connect_rate)(x)
                x = tf.keras.layers.add([x_input, x])
            return x

        def MBConv(x, filters, drop_connect_rate, kernel_size, strides, t, n):
            x = MBConv_idskip(x, filters, drop_connect_rate, kernel_size, strides, t)
            for _ in range(1, n):
                x = MBConv_idskip(x, filters, drop_connect_rate,
                                  kernel_size, strides=1, t=t)
            return x

        num_bands = self._num_bands

        x = conv_bn(allbands, round_filters(32, self.width_coefficient),
                    kernel_size=3, strides=2, activation=True)
        x = MBConv(x, filters=round_filters(16, self.width_coefficient), kernel_size=3,
                   drop_connect_rate=self.dropout_rate, strides=1, t=1, n=round_repeats(1, self.depth_coefficient))
        x = MBConv(x, filters=round_filters(24, self.width_coefficient), kernel_size=3,
                   drop_connect_rate=self.dropout_rate, strides=2, t=6, n=round_repeats(2, self.depth_coefficient))
        x = MBConv(x, filters=round_filters(40, self.width_coefficient), kernel_size=5,
                   drop_connect_rate=self.dropout_rate, strides=2, t=6, n=round_repeats(2, self.depth_coefficient))
        x = MBConv(x, filters=round_filters(80, self.width_coefficient), kernel_size=3,
                   drop_connect_rate=self.dropout_rate, strides=2, t=6, n=round_repeats(3, self.depth_coefficient))
        x = MBConv(x, filters=round_filters(112, self.width_coefficient), kernel_size=5,
                   drop_connect_rate=self.dropout_rate, strides=1, t=6, n=round_repeats(3, self.depth_coefficient))
        x = MBConv(x, filters=round_filters(192, self.width_coefficient), kernel_size=5,
                   drop_connect_rate=self.dropout_rate, strides=2, t=6, n=round_repeats(4, self.depth_coefficient))
        x = MBConv(x, filters=round_filters(320, self.width_coefficient), kernel_size=3,
                   drop_connect_rate=self.dropout_rate, strides=1, t=6, n=round_repeats(1, self.depth_coefficient))

        x = conv_bn(x, filters=round_filters(
            1280, self.width_coefficient), kernel_size=1, strides=1)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

        return x


class ResNet(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):

        def initial_conv(input):
            x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)(input)

            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
            x = tf.keras.layers.Activation('relu')(x)
            return x

        def expand_conv(init, base, k, strides=(1, 1)):
            x = tf.keras.layers.Conv2D(base * k, (3, 3), padding='same', strides=strides,
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)(init)

            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)(x)

            skip = tf.keras.layers.Conv2D(base * k, (1, 1), padding='same', strides=strides,
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          use_bias=False)(init)

            m = Add()([x, skip])
            return m

        def conv1_block(input, k=1, dropout=0.0):
            init = input

            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)(x)

            if dropout > 0.0: x = Dropout(dropout)(x)

            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)(x)

            m = Add()([init, x])
            return m

        def conv2_block(input, k=1, dropout=0.0):
            init = input

            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)(x)

            if dropout > 0.0: x = Dropout(dropout)(x)

            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)(x)

            m = Add()([init, x])
            return m

        def conv3_block(input, k=1, dropout=0.0):
            init = input

            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)(x)

            if dropout > 0.0: x = Dropout(dropout)(x)

            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)(x)

            m = Add()([init, x])
            return m

        depth = 16
        k = 4
        dropout = 0.0
        weight_decay = 0.0005

        N = (depth - 4) // 6

        x = initial_conv(allbands)
        nb_conv = 4

        x = expand_conv(x, 16, k)
        nb_conv += 2

        for i in range(N - 1):
            x = conv1_block(x, k, dropout)
            nb_conv += 2

        x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = expand_conv(x, 32, k, strides=(2, 2))
        nb_conv += 2

        for i in range(N - 1):
            x = conv2_block(x, k, dropout)
            nb_conv += 2

        x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = expand_conv(x, 64, k, strides=(2, 2))
        nb_conv += 2

        for i in range(N - 1):
            x = conv3_block(x, k, dropout)
            nb_conv += 2

        x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.AveragePooling2D((8, 8))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        print('Wide Residual Network-{}-{} created.'.format(nb_conv, k))
        return x


class EfficientNetB0(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None,
            input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class EfficientNetB1(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB1(
            include_top=False, weights=None,
            input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class EfficientNetB2(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB2(
            include_top=False, weights=None,
            input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class EfficientNetB3(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB3(
            include_top=False, weights=None,
            input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class EfficientNetB4(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB4(
            include_top=False, weights=None,
            input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class EfficientNetB5(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB5(
            include_top=False, weights=None,
            input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class EfficientNetB6(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB6(
            include_top=False, weights=None,
            input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class EfficientNetB7(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        num_bands = self._num_bands
        # Add VGG19
        x = tf.keras.applications.EfficientNetB7(
            include_top=False, weights=None,
            input_shape=(120, 120, num_bands), pooling='avg')(allbands)

        return x


class DNN_model(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def fully_connected_block(self, inputs, nb_neurons, name):
        fully_connected_res = tf.keras.layers.Dense(
            units=nb_neurons,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=SEED),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l2(l=2e-5),
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name=name,
        )(inputs)
        batch_res = tf.keras.layers.BatchNormalization()(fully_connected_res)
        return tf.nn.relu(features=batch_res, name='relu')

    def conv_block(self, inputs, nb_filter, filter_size, name):
        conv_res = tf.keras.layers.Conv2D(
            filters=nb_filter,
            kernel_size=filter_size,
            strides=(1, 1),
            padding='same',
            data_format='channels_last',
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=SEED),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l2(l=2e-5),
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name=name
        )(inputs)
        batch_res = tf.keras.layers.BatchNormalization()(conv_res)

        return tf.nn.relu(features=batch_res, name='relu')

    def pooling(self, inputs, name):
        return tf.nn.max_pool(
            input=inputs,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format='NHWC',
            name=name
        )

    def dropout(self, inputs, drop_rate, name):
        return tf.nn.dropout(
            inputs,
            rate=drop_rate,
            noise_shape=None,
            seed=SEED,
            name=name
        )

    def convert_image_to_uint8(self, img_batch):
        return tf.map_fn(
            lambda x: tf.image.convert_image_dtype((tf.image.per_image_standardization(x) + 1.) / 2., dtype=tf.uint8,
                                                   saturate=True), img_batch, dtype=tf.uint8)

    def branch_model_10m(self, inputs):
        out = self.conv_block(inputs, 32, [5, 5], 'conv_block_10_0')
        out = self.pooling(out, 'max_pooling_10')
        out = self.dropout(out, 0.25, 'dropout_10_0')

        out = self.conv_block(out, 32, [5, 5], 'conv_block_10_1')
        out = self.pooling(out, 'max_pooling_10_1')
        out = self.dropout(out, 0.25, 'dropout_10_1')
        out = self.conv_block(out, 64, [3, 3], 'conv_block_10_2')
        out = self.dropout(out, 0.25, 'dropout_10_2')
        out = tf.keras.layers.Flatten()(out)
        out = self.fully_connected_block(out, self.feature_size, 'fc_block_10_0')
        feature = self.dropout(out, 0.5, 'dropout_10_3')
        return feature

    def branch_model_20m(self, inputs):
        out = self.conv_block(inputs, 32, [3, 3], 'conv_block_20_0')
        out = self.pooling(out, 'max_pooling_20_0')
        out = self.dropout(out, 0.25, 'dropout_20_0')
        out = self.conv_block(out, 32, [3, 3], 'conv_block_20_1')
        out = self.dropout(out, 0.25, 'dropout_20_1')
        out = self.conv_block(out, 64, [3, 3], 'conv_block_20_2')
        out = self.dropout(out, 0.25, 'dropout_20_2')
        out = tf.keras.layers.Flatten()(out)
        out = self.fully_connected_block(out, self.feature_size, 'fc_block_20_0')
        feature = self.dropout(out, 0.5, 'dropout_20_3')
        return feature

    def branch_model_60m(self, inputs):
        out = self.conv_block(inputs, 32, [2, 2], 'conv_block_60_0')
        out = self.dropout(out, 0.25, 'dropout_60_0')
        out = self.conv_block(out, 32, [2, 2], 'conv_block_60_1')
        out = self.dropout(out, 0.25, 'dropout_60_1')
        out = self.conv_block(out, 32, [2, 2], 'conv_block_60_2')
        out = self.dropout(out, 0.25, 'dropout_60_2')
        out = tf.keras.layers.Flatten()(out)
        out = self.fully_connected_block(out, self.feature_size, 'fc_block_60_0')
        feature = self.dropout(out, 0.5, 'dropout_60_3')
        return feature

    def _create_model_logits(self, allbands):
        branch_features = []
        for img_bands, nb_bands, branch_model, resolution in zip(
                [self.bands_10m, self.bands_20m, self.bands_60m],
                [self.nb_bands_10m, self.nb_bands_20m, self.nb_bands_60m],
                [self.branch_model_10m, self.branch_model_20m, self.branch_model_60m], ['_10', '_20', '_60']):
            print('Shape : ', img_bands.shape)
            branch_features.append(tf.reshape(branch_model(img_bands), [-1, self.feature_size]))

            patches_concat_embed_ = tf.concat(branch_features, -1)
            patches_concat_embed_ = self.fully_connected_block(patches_concat_embed_, self.feature_size,
                                                               'fc_block_0' + resolution)
            patches_concat_embed_ = self.dropout(patches_concat_embed_, 0.25,
                                                 'dropout_0' + resolution)

        return self.dropout(patches_concat_embed_, 0.5, 'dropout_0' + resolution)




class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Dense(units=projection_dim, use_bias=False)#self.add_weight(
        #    "pos_emb", shape=(1, num_patches + 1, projection_dim)
        #) #tf.keras.layers.Embedding(
        #    input_dim=num_patches, output_dim=projection_dim
        #)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        positions = tf.expand_dims(positions,axis=1)

        emb =  self.position_embedding(positions)
        #print(emb.shape,flush = True)
        proj = self.projection(patch)
        #print('projection shape : ', proj.shape, flush = True)
        encoded = proj + emb
        return encoded

class ViT(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):
        #num_bands = self._num_bands

        #self.learning_rate = 0.001
        #weight_decay = 0.0001
        #batch_size = 256
        #num_epochs = 100
        image_size = 120  # We'll resize input images to this size
        self.patch_size = 6  # Size of the patches to be extract from the input images
        self.num_patches = (image_size // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]  # Size of the transformer layers
        self.transformer_layers = 8
        self.mlp_head_units = [2048, 1024]
        x = self.create_vit_classifier(allbands)

        return x

    def mlp(self,x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def create_vit_classifier(self,allbands):
        num_bands = self._num_bands
        # Augment data.
        #augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(self.patch_size)(allbands)
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        #logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        #model = tf.keras.Model(inputs=allbands, outputs=features)
        #model.summary()
        return features



'''
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ViT(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):

        image_size = 120
        patch_size = 6
        num_layers = 8

        d_model = 64
        num_heads = 4
        mlp_dim = 2048
        channels=12
        dropout=0.1

        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.rescale = Rescaling(1./255)
        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        self.patch_proj = Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dropout(dropout),
                Dense(num_classes),
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x
        
'''