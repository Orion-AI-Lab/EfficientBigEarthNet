#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This script can be used to train any deep learning model on the BigEarthNet. 
#
# To run the code, you need to provide a json file for configurations of the training.
# 
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1
# Usage: train.py [CONFIG_FILE_PATH]

from __future__ import print_function

SEED = 42

import random as rn
rn.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.set_random_seed(SEED)

import os
import argparse
from BigEarthNet import BigEarthNet
from utils import get_metrics
import json
import importlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def gpu_config():
    # GPU configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    #config = tf.ConfigProto(log_device_placement=True)

    # 'Best-fit with coalescing' algorithm for memory allocation
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True

    # Assume data parallelism
    # Pin GPU to be used to process local rank (one GPU per process)
    # config.gpu_options.visible_device_list = str(hvd.local_rank())

    return config


def run_model(args):
    with tf.Session(config=gpu_config()) as sess:
        print('Creating data iterator')
        iterator = BigEarthNet(
            args['tr_tf_record_files'], 
            args['batch_size'], 
            args['nb_epoch'], 
            args['shuffle_buffer_size'],
            args['label_type']
        ).batch_iterator
        nb_iteration = int(np.ceil(float(args['training_size'] * args['nb_epoch']) / args['batch_size']))
        iterator_ins = iterator.get_next()

        print('Creating model: {}'.format(args['model_name']))
        model = importlib.import_module('models.' + args['model_name']).DNN_model(args['label_type'])
        model.create_network()
        loss = model.define_loss()

        print('Creating optimizer and update ops')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=args['learning_rate']).minimize(loss)

        variables_to_save = tf.global_variables()
        _, metric_means, metric_update_ops = get_metrics(model.multi_hot_label, model.predictions, model.probabilities)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        model_saver = tf.train.Saver(max_to_keep=0, var_list=variables_to_save)
        iteration_idx = 0

        if args['fine_tune']:
            model_saver.restore(sess, args['model_file'])
            if 'iteration' in args['model_file']:
                iteration_idx = int(args['model_file'].split('iteration-')[-1])

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(args['out_dir'], 'logs', 'training'), sess.graph)
        
        progress_bar = tf.contrib.keras.utils.Progbar(target = nb_iteration) 
        while True:
            try:
                batch_dict = sess.run(iterator_ins)
            except tf.errors.OutOfRangeError:
                break
            _, _, batch_loss, batch_summary = sess.run([train_op, metric_update_ops, loss, summary_op], 
                                                        feed_dict = model.feed_dict(batch_dict, is_training=True))
            iteration_idx += 1
            summary_writer.add_summary(batch_summary, iteration_idx)
            if (iteration_idx % args['save_checkpoint_per_iteration'] == 0) and (iteration_idx >= args['save_checkpoint_after_iteration']):
                model_saver.save(sess, os.path.join(args['out_dir'], 'models', 'iteration'), iteration_idx)
            progress_bar.update(iteration_idx, values=[('loss', batch_loss)])
        model_saver.save(sess, os.path.join(args['out_dir'], 'models', 'iteration'), iteration_idx)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= 'Training script')
    parser.add_argument('configs', help= 'json config file')
    parser_args = parser.parse_args()

    with open('configs/base.json', 'rb') as f:
        args = json.load(f)

    with open(os.path.realpath(parser_args.configs), 'rb') as f:
        model_args = json.load(f)

    args.update(model_args)
    run_model(args)
