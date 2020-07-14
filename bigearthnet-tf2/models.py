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


MODELS_CLASS = {
    "dense": "BigEarthModel",
    "ResNet50": "ResNet50BigEarthModel",
    "ResNet101": "ResNet101BigEarthModel",
    "ResNet152": "ResNet152BigEarthModel",
    "VGG16": "VGG16BigEarthModel",
    "VGG19": "VGG19BigEarthModel",
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


class ResNet50BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):

        # Use a 1x1 convolution to drop the channels from 12 to 3
        x = Conv2D(
            filters=3,
            kernel_size=(1, 1),
            data_format="channels_last",
            input_shape=(120, 120, 12),
            dtype=self._dtype,
        )(allbands)

        # Add ResNet50 
        x = ResNet50(
            include_top=True,
            weights=None,
            input_shape=(120, 120, 3),
            pooling=max,
        )(x)

        return x


class ResNet101BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):

        # Use a 1x1 convolution to drop the channels from 12 to 3
        x = Conv2D(
            filters=3,
            kernel_size=(1, 1),
            data_format="channels_last",
            input_shape=(120, 120, 12),
            dtype=self._dtype,
        )(allbands)

        # Add ResNet101
        x = ResNet101(
            include_top=True,
            weights=None,
            input_shape=(120, 120, 3),
            pooling=max,
        )(x)

        return x


class ResNet152BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):

        # Use a 1x1 convolution to drop the channels from 12 to 3
        x = Conv2D(
            filters=3,
            kernel_size=(1, 1),
            data_format="channels_last",
            input_shape=(120, 120, 12),
            dtype=self._dtype,
        )(allbands)

        # Add ResNet152
        x = ResNet152(
            include_top=True,
            weights=None,
            input_shape=(120, 120, 3),
            pooling=max,
        )(x)

        return x


class VGG16BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):

        # Use a 1x1 convolution to drop the channels from 12 to 3
        x = Conv2D(
            filters=3,
            kernel_size=(1, 1),
            data_format="channels_last",
            input_shape=(120, 120, 12),
            dtype=self._dtype,
        )(allbands)

        # Add VGG16
        x = VGG16(
            include_top=True,
            weights=None,
            input_shape=(120, 120, 3),
            pooling=max,
            classifier_activation=None,
        )(x)

        return x

class VGG19BigEarthModel(BigEarthModel):
    def __init__(self, nb_class):
        super().__init__(nb_class)

    def _create_model_logits(self, allbands):

        # Use a 1x1 convolution to drop the channels from 12 to 3
        x = Conv2D(
            filters=3,
            kernel_size=(1, 1),
            data_format="channels_last",
            input_shape=(120, 120, 12),
            dtype=self._dtype,
        )(allbands)

        # Add VGG19
        x = VGG19(
            include_top=True,
            weights=None,
            input_shape=(120, 120, 3),
            pooling=max,
            classifier_activation=None,
        )(x)

        return x        