#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# TODO
#
import tensorflow as tf
import random as rn
import numpy as np
import argparse
import os
import sys
import json

from inputs import create_batched_dataset
from model import BigEarthModel

SEED = 42


def run_model(args):
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("Running using random seed: {}".format(SEED))

    rn.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Create our data pipeline
    batched_dataset = create_batched_dataset(
        args["tr_tf_record_files"],
        args["batch_size"],
        args["nb_epoch"],
        args["shuffle_buffer_size"],
        args["label_type"],
    )

    # Create our model
    bigearth_model = BigEarthModel(label_type=args["label_type"])
    model = bigearth_model.model

    # DEBUG
    # single_batch = next(iter(batched_dataset))
    # x = single_batch["B01"]
    # y = single_batch[args["label_type"] + "_labels_multi_hot"]
    # y_ = model(x, training=True)
    # print(y)
    # print(y_)

    # Create loss
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    def loss(model, x, y, training):
        y_ = model(x, training=training)
        return loss_object(y_true=y, y_pred=y_)

    # Setup gradients
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    train_precision_results = []
    train_recall_results = []

    # The main loop
    batch_size = args["batch_size"]
    for epoch in range(args["nb_epoch"]):
        print("Starting epoch {}".format(epoch))

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        epoch_precision = tf.keras.metrics.Precision()
        epoch_recall = tf.keras.metrics.Recall()

        batch_iterator = iter(batched_dataset)

        for single_batch in batch_iterator:
            x = single_batch["B01"]
            y = single_batch[args["label_type"] + "_labels_multi_hot"]

            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            y_ = model(x, training=True)
            epoch_accuracy.update_state(y, y_)
            epoch_precision.update_state(y, y_)
            epoch_recall.update_state(y, y_)

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        train_precision_results.append(epoch_precision.result())
        train_recall_results.append(epoch_recall.result())

        # if epoch % 5 == 0:
        print(
            "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3%}, Recall: {:.3%}".format(
                epoch,
                epoch_loss_avg.result(),
                epoch_accuracy.result(),
                epoch_precision.result(),
                epoch_recall.result(),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("configs", help="json config file")
    parser_args = parser.parse_args()

    with open("configs/base.json", "rb") as f:
        args = json.load(f)

    with open(os.path.realpath(parser_args.configs), "rb") as f:
        model_args = json.load(f)
        args.update(model_args)

    run_model(args)

