# -*- coding: utf-8 -*-
#
# The models
#

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Lambda, Conv2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, VGG16, VGG19

from tensorflow.python.ops import nn

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
    "DenseNet201": "DenseNet201BigEarthModel"
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
        #allbands = tf.concat([bands_10m, bands_20m], axis=3)
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
            include_top=False, weights=None, input_shape=(120,120,num_bands))(allbands)

        return x

class DenseNet169BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):

        num_bands = self._num_bands
        x = tf.keras.applications.DenseNet169(
            include_top=False, weights=None, input_shape=(120,120,num_bands))(allbands)

        return x


class DenseNet201BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):

        num_bands = self._num_bands
        x = tf.keras.applications.DenseNet201(
            include_top=False, weights=None, input_shape=(120,120,num_bands))(allbands)

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
                name = name
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
