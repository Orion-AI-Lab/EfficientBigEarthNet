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
from metrics import CustomMetrics
from models import MODELS_CLASS
import models

SEED = 42


def evaluate_model(model, batched_dataset, nb_class):
    """Evaluate model using the eval dataset
    """
    print('Running model on validation dataset')
    custom_metrics = CustomMetrics(nb_class=nb_class)

    for i, single_batch in enumerate(iter(batched_dataset)):
        x_all = [
            single_batch["B01"],
            single_batch["B02"],
            single_batch["B03"],
            single_batch["B04"],
            single_batch["B05"],
            single_batch["B06"],
            single_batch["B07"],
            single_batch["B08"],
            single_batch["B8A"],
            single_batch["B09"],
            single_batch["B11"],
            single_batch["B12"],
        ]
        y = single_batch[args["label_type"] + "_labels_multi_hot"]

        # Compare predicted label to actual label
        y_ = model(x_all, training=False)
        # Update all custom metrics
        custom_metrics.update_state(y, y_)

    return custom_metrics.result()


def run_model(args):
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("Running using random seed: {}".format(SEED))

    print("Batch size: {}".format(args["batch_size"]))
    print("Epochs: {}".format(args["nb_epoch"]))
    if args['parallel'] == False:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for d in physical_devices:
            tf.config.experimental.set_memory_growth(d, True)

    rn.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Create our data pipeline
    train_batched_dataset = create_batched_dataset(
        args["tr_tf_record_files"],
        args["batch_size"],
        args["shuffle_buffer_size"],
        args["label_type"],
        args['num_workers'],
        args['worker_index'],
    )

    val_batched_dataset = create_batched_dataset(
        args["val_tf_record_files"],
        args["batch_size"],
        args["shuffle_buffer_size"],
        args["label_type"],
        args['num_workers'],
        args['worker_index'],
    )

    # Create our model
    nb_class = 19 if args["label_type"] == "BigEarthNet-19" else 43

    try:
        bigearth_model_class = MODELS_CLASS[args["model_name"]]
    except:
        bigearth_model_class = MODELS_CLASS["dense"]

    print('Creating model: {}'.format(args['model_name']))
    bigearth_model = getattr(models, bigearth_model_class)(nb_class=nb_class)
    model = bigearth_model.model

    # DEBUG (use this to understand what the iterators are returning)
    debug = False
    if debug:
        single_batch = next(iter(train_batched_dataset))
        x_all = [
            single_batch["B01"],
            single_batch["B02"],
            single_batch["B03"],
            single_batch["B04"],
            single_batch["B05"],
            single_batch["B06"],
            single_batch["B07"],
            single_batch["B08"],
            single_batch["B8A"],
            single_batch["B09"],
            single_batch["B11"],
            single_batch["B12"],
        ]
        y = single_batch[args["label_type"] + "_labels_multi_hot"]
        y_ = model(x_all, training=True)
        # print(x_all)
        print(y)
        print(y_)

    # Create loss
    loss_object = tf.keras.losses.BinaryCrossentropy()

    def loss(model, x, y, training):
        y_ = model(x, training=training)
        return loss_object(y_true=y, y_pred=y_)

    # Setup gradients
    @tf.function
    def grad(model, inputs, targets, first_batch):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        if args['parallel']:
            # Horovod: add Horovod Distributed GradientTape.
            tape = hvd.DistributedGradientTape(tape)
            if first_batch:
                hvd.broadcast_variables(model.variables, root_rank=0)
                hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args["learning_rate"] * args["num_workers"])

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    train_precision_results = []
    train_recall_results = []

    # The main loop
    batch_size = args["batch_size"]
    epoch_custom_metrics = CustomMetrics(nb_class=nb_class)
    for epoch in range(args["nb_epoch"]):
        print("Starting epoch {}".format(epoch))

        epoch_loss_avg = tf.keras.metrics.Mean(dtype='float64')
        epoch_custom_metrics.reset_states()

        nb_iterations = args["training_size"] / args["batch_size"]
        if args["training_size"] % args["batch_size"] != 0:
            nb_iterations += 1
        progress_bar = tf.keras.utils.Progbar(target=nb_iterations)

        batch_iterator = iter(train_batched_dataset)
        for i, single_batch in enumerate(batch_iterator):
            x_all = [
                single_batch["B01"],
                single_batch["B02"],
                single_batch["B03"],
                single_batch["B04"],
                single_batch["B05"],
                single_batch["B06"],
                single_batch["B07"],
                single_batch["B08"],
                single_batch["B8A"],
                single_batch["B09"],
                single_batch["B11"],
                single_batch["B12"],
            ]
            y = single_batch[args["label_type"] + "_labels_multi_hot"]

            # Optimize the model
            first_batch = (i == 0 and epoch == 0)
            loss_value, grads = grad(model, x_all, y, first_batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            y_ = model(x_all, training=False)
            # Update all custom metrics
            epoch_custom_metrics.update_state(y, y_)

            progress_bar.update(i + 1)

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        flag = False
        if args["parallel"]:
            if hvd.rank() == 0:
                flag = True
        else:
            flag = True

        if flag:
            # if epoch>5:
            print(" Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result(), ))
            # Evaluate model using the eval dataset
            (
                epoch_micro_precision,
                epoch_macro_precision,
                epoch_micro_recall,
                epoch_macro_recall,
                epoch_micro_accuracy,
                epoch_macro_accuracy,
                f_score
            ) = evaluate_model(model, val_batched_dataset, nb_class)

            print(
                "Epoch {:03d}: micro: accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, F-score: {:.3f}".format(
                    epoch, epoch_micro_accuracy, epoch_micro_precision, epoch_micro_recall, f_score
                )
            )
            print(
                "Epoch {:03d}: macro: accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}".format(
                    epoch, epoch_macro_accuracy, epoch_macro_precision, epoch_macro_recall
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--configs", required=False, default='configs/base.json', help="JSON config file")
    parser.add_argument("--parallel", required=False, default=False, help="Enable parallelism")
    parser_args = parser.parse_args()

    with open(parser_args.configs, "rb") as f:
        args = json.load(f)
        args.update(vars(parser_args))

    if parser_args.parallel:
        import horovod.tensorflow as hvd
        hvd.init()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        args['num_workers'] = hvd.size()
        args['worker_index'] = hvd.rank()
    else:
        args['num_workers'] = 1
        args['worker_index'] = None

    run_model(args)