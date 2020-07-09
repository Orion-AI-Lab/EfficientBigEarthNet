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

from tensorflow.keras.layers import Input, Dense, Flatten, Activation
from tensorflow.keras.models import Model

# from tensorflow.keras.applications import ResNet50

from inputs import BAND_STATS


class BigEarthModel:
    def __init__(self, label_type):
        self.label_type = label_type
        self.nb_class = 19 if label_type == "BigEarthNet-19" else 43
        # self.prediction_threshold = 0.5

        self._inputB01 = Input(shape=(20, 20,), dtype=tf.float32)
        self._inputB09 = Input(shape=(20, 20,), dtype=tf.float32)
        bands_20 = tf.keras.backend.stack([self._inputB01, self._inputB09], axis=3)
        print(bands_20.shape)

        # TODO: concat, stack and combine all inputs
        # TODO: then create some nice model
        x = Flatten()(bands_20)
        x = Dense(64)(x)
        x = Dense(19)(x)

        self._logits = x
        self._output = Activation("sigmoid")(x)
        self._model = Model(inputs=[self._inputB01, self._inputB09], outputs=self._output)
        self._logits_model = Model(inputs=[self._inputB01, self._inputB09], outputs=self._logits)

    @property
    def model(self):
        return self._model

    @property
    def logits_model(self):
        return self._logits_model


class OldModel:
    def __init__(self, label_type):
        self.label_type = label_type
        self.prediction_threshold = 0.5
        self.is_training = tf.placeholder(tf.bool, [])
        self.nb_class = 19 if label_type == "BigEarthNet-19" else 43

        self.B04 = tf.placeholder(tf.float32, [None, 120, 120], name="B04")
        self.B03 = tf.placeholder(tf.float32, [None, 120, 120], name="B03")
        self.B02 = tf.placeholder(tf.float32, [None, 120, 120], name="B02")
        self.B08 = tf.placeholder(tf.float32, [None, 120, 120], name="B08")
        self.bands_10m = tf.stack([self.B04, self.B03, self.B02, self.B08], axis=3)

        self.B05 = tf.placeholder(tf.float32, [None, 60, 60], name="B05")
        self.B06 = tf.placeholder(tf.float32, [None, 60, 60], name="B06")
        self.B07 = tf.placeholder(tf.float32, [None, 60, 60], name="B07")
        self.B8A = tf.placeholder(tf.float32, [None, 60, 60], name="B8A")
        self.B11 = tf.placeholder(tf.float32, [None, 60, 60], name="B11")
        self.B12 = tf.placeholder(tf.float32, [None, 60, 60], name="B12")
        self.bands_20m = tf.stack(
            [self.B05, self.B06, self.B07, self.B8A, self.B11, self.B12], axis=3
        )

        self.B01 = tf.placeholder(tf.float32, [None, 20, 20], name="B01")
        self.B09 = tf.placeholder(tf.float32, [None, 20, 20], name="B09")
        self.bands_60m = tf.stack([self.B01, self.B09], axis=3)

        self.img = tf.concat(
            [self.bands_10m, tf.image.resize_bicubic(self.bands_20m, [120, 120])],
            axis=3,
        )
        self.multi_hot_label = tf.placeholder(tf.float32, shape=(None, self.nb_class))
        self.model_path = tf.placeholder(tf.string)

    def define_loss(self):
        self.loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=self.multi_hot_label, logits=self.logits
        )
        tf.summary.scalar("sigmoid_cross_entropy_loss", self.loss)
        return self.loss


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
