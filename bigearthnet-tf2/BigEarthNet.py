# -*- coding: utf-8 -*-
#
# BigEarthNet class to create tf.data.Dataset based on the TFRecord files. 
#
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1

import tensorflow as tf

BAND_STATS = {
            'mean': {
                'B01': 340.76769064,
                'B02': 429.9430203,
                'B03': 614.21682446,
                'B04': 590.23569706,
                'B05': 950.68368468,
                'B06': 1792.46290469,
                'B07': 2075.46795189,
                'B08': 2218.94553375,
                'B8A': 2266.46036911,
                'B09': 2246.0605464,
                'B11': 1594.42694882,
                'B12': 1009.32729131
            },
            'std': {
                'B01': 554.81258967,
                'B02': 572.41639287,
                'B03': 582.87945694,
                'B04': 675.88746967,
                'B05': 729.89827633,
                'B06': 1096.01480586,
                'B07': 1273.45393088,
                'B08': 1365.45589904,
                'B8A': 1356.13789355,
                'B09': 1302.3292881,
                'B11': 1079.19066363,
                'B12': 818.86747235
            }
        }


def parse_function(example_proto, label_type):
    nb_class = 43 if label_type == 'original' else 19

    parsed_features = tf.io.parse_single_example(
            example_proto, 
            {
                'B01': tf.io.FixedLenFeature([20*20], tf.int64),
                'B02': tf.io.FixedLenFeature([120*120], tf.int64),
                'B03': tf.io.FixedLenFeature([120*120], tf.int64),
                'B04': tf.io.FixedLenFeature([120*120], tf.int64),
                'B05': tf.io.FixedLenFeature([60*60], tf.int64),
                'B06': tf.io.FixedLenFeature([60*60], tf.int64),
                'B07': tf.io.FixedLenFeature([60*60], tf.int64),
                'B08': tf.io.FixedLenFeature([120*120], tf.int64),
                'B8A': tf.io.FixedLenFeature([60*60], tf.int64),
                'B09': tf.io.FixedLenFeature([20*20], tf.int64),
                'B11': tf.io.FixedLenFeature([60*60], tf.int64),
                'B12': tf.io.FixedLenFeature([60*60], tf.int64),
                label_type + '_labels': tf.io.VarLenFeature(dtype=tf.string),
                label_type + '_labels_multi_hot': tf.io.FixedLenFeature([nb_class], tf.int64),
                'patch_name': tf.io.VarLenFeature(dtype=tf.string)
            }
        )

    return {
        'B01': tf.reshape(parsed_features['B01'], [20, 20]),
        'B02': tf.reshape(parsed_features['B02'], [120, 120]),
        'B03': tf.reshape(parsed_features['B03'], [120, 120]),
        'B04': tf.reshape(parsed_features['B04'], [120, 120]),
        'B05': tf.reshape(parsed_features['B05'], [60, 60]),
        'B06': tf.reshape(parsed_features['B06'], [60, 60]),
        'B07': tf.reshape(parsed_features['B07'], [60, 60]),
        'B08': tf.reshape(parsed_features['B08'], [120, 120]),
        'B8A': tf.reshape(parsed_features['B8A'], [60, 60]),
        'B09': tf.reshape(parsed_features['B09'], [20, 20]),
        'B11': tf.reshape(parsed_features['B11'], [60, 60]),
        'B12': tf.reshape(parsed_features['B12'], [60, 60]),
        label_type + '_labels': parsed_features[label_type + '_labels'],
        label_type + '_labels_multi_hot': parsed_features[label_type + '_labels_multi_hot'],
        'patch_name': parsed_features['patch_name']
    }


def input_fn(TFRecord_paths, batch_size, nb_epoch, shuffle_buffer_size, label_type):
    dataset = tf.data.TFRecordDataset(TFRecord_paths)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.repeat(nb_epoch)

    dataset = dataset.map(
        lambda x: parse_function(x, label_type), 
        num_parallel_calls=10
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)
    batched_dataset = dataset.prefetch(10)
    return iter(batched_dataset)

