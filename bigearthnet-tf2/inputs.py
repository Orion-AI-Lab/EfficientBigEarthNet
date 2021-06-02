# -*- coding: utf-8 -*-
#
# BigEarthNet class to create tf.data.Dataset based on the TFRecord files.
#
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1

import tensorflow as tf
import numpy as np
from glob import glob

BAND_STATS = {
    "mean": {
        "B02": 429.9430203,
        "B03": 614.21682446,
        "B04": 590.23569706,
        "B05": 950.68368468,
        "B06": 1792.46290469,
        "B07": 2075.46795189,
        "B08": 2218.94553375,
        "B8A": 2266.46036911,
        "B11": 1594.42694882,
        "B12": 1009.32729131,
    },
    "std": {
        "B02": 572.41639287,
        "B03": 582.87945694,
        "B04": 675.88746967,
        "B05": 729.89827633,
        "B06": 1096.01480586,
        "B07": 1273.45393088,
        "B08": 1365.45589904,
        "B8A": 1356.13789355,
        "B11": 1079.19066363,
        "B12": 818.86747235,
    },
}


def _parse_function(example_proto, label_type):
    """Parse an example from a tfrecord.
    """
    nb_class = 43 if label_type == "original" else 19

    parsed_features = tf.io.parse_single_example(
        example_proto,
        {
            "B02": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B03": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B04": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B05": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B06": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B07": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B08": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B8A": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B11": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B12": tf.io.FixedLenFeature([60 * 60], tf.int64),
            label_type + "_labels": tf.io.VarLenFeature(dtype=tf.string),
            label_type
            + "_labels_multi_hot": tf.io.FixedLenFeature([nb_class], tf.int64),
            "patch_name": tf.io.VarLenFeature(dtype=tf.string),
        },
    )

    return {
        "B02": tf.cast(tf.reshape(parsed_features["B02"], [120, 120]), tf.float32),
        "B03": tf.cast(tf.reshape(parsed_features["B03"], [120, 120]), tf.float32),
        "B04": tf.cast(tf.reshape(parsed_features["B04"], [120, 120]), tf.float32),
        "B05": tf.cast(tf.reshape(parsed_features["B05"], [60, 60]), tf.float32),
        "B06": tf.cast(tf.reshape(parsed_features["B06"], [60, 60]), tf.float32),
        "B07": tf.cast(tf.reshape(parsed_features["B07"], [60, 60]), tf.float32),
        "B08": tf.cast(tf.reshape(parsed_features["B08"], [120, 120]), tf.float32),
        "B8A": tf.cast(tf.reshape(parsed_features["B8A"], [60, 60]), tf.float32),
        "B11": tf.cast(tf.reshape(parsed_features["B11"], [60, 60]), tf.float32),
        "B12": tf.cast(tf.reshape(parsed_features["B12"], [60, 60]), tf.float32),
        label_type + "_labels": parsed_features[label_type + "_labels"],
        label_type
        + "_labels_multi_hot": parsed_features[label_type + "_labels_multi_hot"],
        "patch_name": parsed_features["patch_name"],
    }


def _preprocess_function(example, label_type):
    """Preprocess an example.
    """

    newexample = {}
    for band in BAND_STATS["mean"].keys():
        band_tensor = example[band]
        newexample[band] = tf.divide(
            tf.math.subtract(
                band_tensor,
                tf.constant(BAND_STATS["mean"][band], shape=band_tensor.shape),
            ),
            tf.constant(BAND_STATS["std"][band], shape=band_tensor.shape),
        )

        # TODO: add some tranformations here, like flip left-right, small rotation, etc.

    newexample[label_type + "_labels"] = example[label_type + "_labels"]
    newexample[label_type + "_labels_multi_hot"] = example[
        label_type + "_labels_multi_hot"
    ]
    newexample["patch_name"] = example["patch_name"]

    return newexample


def create_batched_dataset(TFRecord_paths, batch_size, shuffle_buffer_size, label_type, num_workers, worker_index,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE):
    """Create an input batched dataset.
    """
    parse_fn = lambda x: _parse_function(x, label_type=label_type)
    preprocess_fn = lambda x: _preprocess_function(x, label_type=label_type)

    records = []
    for i in TFRecord_paths:
        records.extend(glob(i))

    dataset = tf.data.Dataset.list_files(TFRecord_paths)
    dataset = dataset.shard(num_workers, worker_index)
    dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=num_parallel_calls)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=num_parallel_calls)

    dataset = dataset.batch(batch_size, drop_remainder=False)

    batched_dataset = dataset.prefetch(num_parallel_calls)
    return batched_dataset
