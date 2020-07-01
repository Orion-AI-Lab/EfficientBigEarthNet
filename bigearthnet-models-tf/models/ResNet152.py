# -*- coding: utf-8 -*-
#
# ResNet152 model definition is based on TensorFlow Slim implementation. 
#
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1

import tensorflow as tf
from nets.resnet_utils import resnet_arg_scope
from nets.resnet_v1 import resnet_v1_152
from models.main_model import Model

class DNN_model(Model):
    def create_network(self):
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            logits, end_points = resnet_v1_152(
                self.img,
                num_classes = self.nb_class,
                is_training = self.is_training,
                global_pool=True,
                spatial_squeeze=True
            )
        self.logits = logits
        self.probabilities = tf.nn.sigmoid(self.logits)
        self.predictions = tf.cast(self.probabilities >= self.prediction_threshold, tf.float32)
