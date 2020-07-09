# -*- coding: utf-8 -*-
#
# Main model to define initial placeholders, feed dictionary, and image normalization.
#
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

from inputs import BAND_STATS


MODELS_CLASS = {
    "ResNet50": "ResNet50BigEarthModel",
}


class BigEarthModel:
    def __init__(self, label_type):
        self.label_type = label_type
        self._nb_class = 19 if label_type == "BigEarthNet-19" else 43
        # self.prediction_threshold = 0.5

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

        # create model
        self._logits = self._create_model_logits(allbands)
        self._output = Activation("sigmoid")(self._logits)
        self._model = Model(inputs=inputs, outputs=self._output)
        self._logits_model = Model(inputs=inputs, outputs=self._logits)

    @property
    def model(self):
        return self._model

    @property
    def logits_model(self):
        return self._logits_model

    def _create_model_logits(self, allbands):
        x = Flatten()(allbands)
        x = Dense(64)(x)
        x = Dense(self._nb_class)(x)
        return x


class ResNet50BigEarthModel(BigEarthModel):
    def __init__(self, label_type):
        super().__init__(label_type)

    def _create_model_logits(self, allbands):

        # Use a 1x1 convolution to drop the channels from 12 to 3
        x = Conv2D(
            filters=3,
            kernel_size=(1, 1),
            data_format="channels_last",
            input_shape=(120, 120, 12),
        )(allbands)

        # Add ResNet50 with additional dense layer as the end
        x = ResNet50(
            include_top=True,
            weights=None,
            input_shape=(120, 120, 3),
            pooling=max,
            classes=self._nb_class,
        )(x)

        return x


#    def create_network(self):
#        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
#            logits, end_points = resnet_v1_50(
#                self.img,
#                num_classes = self.nb_class,
#                is_training = self.is_training,
#                global_pool=True,
#                spatial_squeeze=True
#            )
#        self.logits = logits
#        self.probabilities = tf.nn.sigmoid(self.logits)
#        self.predictions = tf.cast(self.probabilities >= self.prediction_threshold, tf.float32)
