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
import json

from BigEarthNet import input_fn

SEED = 42


def run_model(args):
    rn.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("Running using random seed: {}".format(SEED))

    print("Creating data iterator")
    iterator = input_fn(
        args["tr_tf_record_files"],
        args["batch_size"],
        args["nb_epoch"],
        args["shuffle_buffer_size"],
        args["label_type"],
    )

    nb_iteration = int(
        np.ceil(float(args["training_size"] * args["nb_epoch"]) / args["batch_size"])
    )
    iterator_ins = iterator.get_next()

    # TODO: create model and training loop


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

