# -*- coding: utf-8 -*-
#
# K-Branch CNN is proposed as the first module of the following system without an attention mechanism:
#
# G. Sumbul, B. Demir, "A Novel Multi-Attention Driven System for Multi-Label Remote Sensing Image Classification", 
# IEEE International Conference on Geoscience and Remote Sensing Symposium, Yokohama, Japan, 2019.
#
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1

import tensorflow as tf
from models.main_model import Model

SEED = 42

class DNN_model(Model):
    def __init__(self, label_type):
        Model.__init__(self, label_type)
        self.feature_size = 128
        self.nb_bands_10m = 4
        self.nb_bands_20m = 6
        self.nb_bands_60m = 2

    def fully_connected_block(self, inputs, nb_neurons, is_training, name):
        with tf.variable_scope(name):
            fully_connected_res = tf.layers.dense(
                inputs = inputs,
                units = nb_neurons,
                activation = None,
                use_bias = True,
                kernel_initializer= tf.contrib.layers.xavier_initializer_conv2d(seed = SEED),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(2e-5),
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name = 'fc_layer',
                reuse=None
            )
            batch_res = tf.layers.batch_normalization(inputs = fully_connected_res, name = 'batch_norm', training = is_training)
            return tf.nn.relu(features = batch_res, name = 'relu')

    def conv_block(self, inputs, nb_filter, filter_size, is_training, name):
        with tf.variable_scope(name):
            conv_res = tf.layers.conv2d(
                inputs = inputs,
                filters = nb_filter,
                kernel_size = filter_size,
                strides = (1, 1),
                padding = 'same',
                data_format = 'channels_last',
                dilation_rate = (1, 1),
                activation = None,
                use_bias = True,
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(seed = SEED),
                bias_initializer = tf.zeros_initializer(),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(2e-5),
                bias_regularizer = None,
                activity_regularizer = None,
                trainable = True,
                name = 'conv_layer',
                reuse = None
            )
            batch_res = tf.layers.batch_normalization(inputs = conv_res, name = 'batch_norm', training = is_training)
            return tf.nn.relu(features = batch_res, name = 'relu')

    def pooling(self, inputs, name):
        return tf.nn.max_pool(
            value = inputs,
            ksize = [1, 2, 2, 1],
            strides = [1, 2, 2, 1],
            padding = "VALID",
            data_format ='NHWC',
            name = name
        )

    def dropout(self, inputs, drop_rate, is_training, name):
        return tf.layers.dropout(
                inputs,
                rate = drop_rate,
                noise_shape = None,
                seed = SEED,
                training = is_training,
                name = name
            )

    def convert_image_to_uint8(self, img_batch):
        return tf.map_fn(lambda x: tf.image.convert_image_dtype((tf.image.per_image_standardization(x) + 1.) / 2., dtype=tf.uint8, saturate=True), img_batch, dtype=tf.uint8)

    def branch_model_10m(self, inputs, is_training):
        with tf.variable_scope('CNN_10m_branch'):
            out = self.conv_block(inputs, 32, [5,5], is_training, 'conv_block_0')
            out = self.pooling(out, 'max_pooling')
            out = self.dropout(out, 0.25, is_training, 'dropout_0')
            out = self.conv_block(out, 32, [5,5], is_training, 'conv_block_1')
            out = self.pooling(out, 'max_pooling_1')
            out = self.dropout(out, 0.25, is_training, 'dropout_1')
            out = self.conv_block(out, 64, [3,3], is_training, 'conv_block_2')
            out = self.dropout(out, 0.25, is_training, 'dropout_2')
            out = tf.contrib.layers.flatten(out)
            out = self.fully_connected_block(out, self.feature_size, is_training, 'fc_block_0')
            feature = self.dropout(out, 0.5, is_training, 'dropout_3')
            return feature 

    def branch_model_20m(self, inputs, is_training):
        with tf.variable_scope('CNN_20m_branch'):
            out = self.conv_block(inputs, 32, [3,3], is_training, 'conv_block_0')
            out = self.pooling(out, 'max_pooling_0')
            out = self.dropout(out, 0.25, is_training, 'dropout_0')
            out = self.conv_block(out, 32, [3,3], is_training, 'conv_block_1')
            out = self.dropout(out, 0.25, is_training, 'dropout_1')
            out = self.conv_block(out, 64, [3,3], is_training, 'conv_block_2')
            out = self.dropout(out, 0.25, is_training, 'dropout_2')
            out = tf.contrib.layers.flatten(out)
            out = self.fully_connected_block(out, self.feature_size, is_training, 'fc_block_0')
            feature = self.dropout(out, 0.5, is_training, 'dropout_3')  
            return feature

    def branch_model_60m(self, inputs, is_training):
        with tf.variable_scope('CNN_60m_branch'):
            out = self.conv_block(inputs, 32, [2,2], is_training, 'conv_block_0')
            out = self.dropout(out, 0.25, is_training, 'dropout_0')
            out = self.conv_block(out, 32, [2,2], is_training, 'conv_block_1')
            out = self.dropout(out, 0.25, is_training, 'dropout_1')
            out = self.conv_block(out, 32, [2,2], is_training, 'conv_block_2')
            out = self.dropout(out, 0.25, is_training, 'dropout_2')
            out = tf.contrib.layers.flatten(out)
            out = self.fully_connected_block(out, self.feature_size, is_training, 'fc_block_0')
            feature = self.dropout(out, 0.5, is_training, 'dropout_3') 
            return feature

    def create_network(self): 
        branch_features = []
        for img_bands, nb_bands, branch_model in zip(
            [self.bands_10m, self.bands_20m, self.bands_60m], 
            [self.nb_bands_10m, self.nb_bands_20m, self.nb_bands_60m],
            [self.branch_model_10m, self.branch_model_20m, self.branch_model_60m]
            ):
            branch_features.append(tf.reshape(branch_model(img_bands, self.is_training), [-1, self.feature_size]))

        with tf.variable_scope('feature_fusion'):
            patches_concat_embed_ = tf.concat(branch_features, -1)
            patches_concat_embed_ = self.fully_connected_block(patches_concat_embed_, self.feature_size, self.is_training, 'fc_block_0')
            patches_concat_embed_ = self.dropout(patches_concat_embed_, 0.25, self.is_training, 'dropout_0')
        
        initializer = tf.contrib.layers.xavier_initializer(seed = SEED)
        with tf.variable_scope('classification'):
            res = self.dropout(patches_concat_embed_, 0.5, self.is_training, 'dropout_0')
            self.logits = tf.layers.dense(
                inputs = res,
                units = self.nb_class,
                activation=None,
                use_bias=True,
                kernel_initializer=initializer,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name='fc_layer',
                reuse=None
            )
            self.probabilities = tf.nn.sigmoid(self.logits)
            self.predictions = tf.cast(self.probabilities >= self.prediction_threshold, tf.float32)
