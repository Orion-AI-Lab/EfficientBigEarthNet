# -*- coding: utf-8 -*-
#
# Utility functions for training and evaluation.
# 
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1

import tensorflow as tf

def get_metrics(labels, predictions, probabilities):
    labels = tf.cast(labels, tf.float64)
    predictions = tf.cast(predictions, tf.float64)
    probabilities = tf.cast(probabilities, tf.float64)
    nb_ins = tf.shape(labels)[0]
    nb_class = tf.shape(labels)[1]
    with tf.variable_scope('metrics'):
        metric_update_ops = []
        zero = tf.constant(0., dtype=tf.float64)
        true_positive = tf.cast(tf.logical_and(tf.not_equal(predictions, zero), tf.not_equal(labels, zero)), tf.float64)
        false_positive = tf.cast(tf.logical_and(tf.not_equal(predictions, zero), tf.equal(labels, zero)), tf.float64)
        true_negative = tf.cast(tf.logical_and(tf.equal(predictions, zero), tf.equal(labels, zero)), tf.float64)
        false_negative = tf.cast(tf.logical_and(tf.equal(predictions, zero), tf.not_equal(labels, zero)), tf.float64)
        label_union_prediction = tf.cast(tf.logical_or(tf.not_equal(predictions, zero), tf.not_equal(labels, zero)), tf.float64) 
        label_cardinality = tf.reduce_sum(labels, 1)

        tp_class_accums = []
        for i in tf.unstack(true_positive, axis=1):
            init_class_tp = tf.Variable(0., dtype=tf.float64)
            metric_update_ops.append(tf.assign_add(init_class_tp, tf.reduce_sum(i)))
            tp_class_accums.append(init_class_tp)

        fp_class_accums = []
        for i in tf.unstack(false_positive, axis=1):
            init_class_fp = tf.Variable(0., dtype=tf.float64)
            metric_update_ops.append(tf.assign_add(init_class_fp, tf.reduce_sum(i)))
            fp_class_accums.append(init_class_fp)

        tn_class_accums = []
        for i in tf.unstack(true_negative, axis=1):
            init_class_tn = tf.Variable(0., dtype=tf.float64)
            metric_update_ops.append(tf.assign_add(init_class_tn, tf.reduce_sum(i)))
            tn_class_accums.append(init_class_tn)

        fn_class_accums = []
        for i in tf.unstack(false_negative, axis=1):
            init_class_fn = tf.Variable(0., dtype=tf.float64)
            metric_update_ops.append(tf.assign_add(init_class_fn, tf.reduce_sum(i)))
            fn_class_accums.append(init_class_fn)

        label_union_pred_class_accums = []
        for i in tf.unstack(label_union_prediction, axis=1):
            init_class_label_union_pred = tf.Variable(0., dtype=tf.float64)
            metric_update_ops.append(tf.assign_add(init_class_label_union_pred, tf.reduce_sum(i)))
            label_union_pred_class_accums.append(init_class_label_union_pred)

        # Precision
        sample_precision = tf.where(tf.equal(tf.reduce_sum(predictions, 1), zero),
                    x = tf.zeros(dtype=tf.float64, shape=nb_ins),
                    y = tf.divide(tf.reduce_sum(true_positive, 1), tf.reduce_sum(predictions, 1)))
        
        with tf.control_dependencies(metric_update_ops):
            nb_label = tf.add_n(tp_class_accums) + tf.add_n(fp_class_accums)
            micro_precision = tf.where(
                tf.equal(nb_label, zero),
                x = zero,
                y = tf.divide(tf.add_n(tp_class_accums), nb_label)
            )

        with tf.control_dependencies(metric_update_ops):
            macro_precision_class = []
            for tp_class, fp_class in zip(tp_class_accums, fp_class_accums):
                nb_predict_class = tp_class + fp_class
                macro_precision_class.append(
                        tf.where(tf.equal(nb_predict_class, zero), x = zero, y = tf.divide(tp_class, nb_predict_class))
                    )
            macro_precision_class_avg = tf.reduce_mean(tf.stack(macro_precision_class))

        # Recall
        sample_recall = tf.where(tf.equal(label_cardinality, zero),
                    x = tf.zeros(dtype=tf.float64, shape=nb_ins),
                    y = tf.divide(tf.reduce_sum(true_positive, 1), label_cardinality))

        with tf.control_dependencies(metric_update_ops):
            nb_label = tf.add_n(tp_class_accums) + tf.add_n(fn_class_accums)
            micro_recall = tf.where(
                    tf.equal(nb_label, zero),
                    x = zero,
                    y = tf.divide(tf.add_n(tp_class_accums), nb_label)
                )

        with tf.control_dependencies(metric_update_ops):
            macro_recall_class = []
            for tp_class, fn_class in zip(tp_class_accums, fn_class_accums):
                nb_label_class = tp_class + fn_class
                macro_recall_class.append(
                    tf.where(tf.equal(nb_label_class, zero), x = zero, y = tf.divide(tp_class, nb_label_class))
                )
            macro_recall_class_avg = tf.reduce_mean(tf.stack(macro_recall_class))

        # F-Score
        fbeta_score = lambda beta, precision, recall: tf.where(
            tf.equal(((beta * beta) * precision) + recall, zero),
            x = tf.zeros(dtype=tf.float64, shape=tf.shape(precision)),
            y = tf.divide((1. + (beta * beta)) * tf.multiply(precision, recall), ((beta * beta) * precision) + recall)
        )

        fbeta_score_core = lambda beta, tp, fp, tn, fn: tf.where(
            tf.equal(((1. + (beta * beta)) * tp )+ ((beta * beta) * fn) + fp, zero),
            x = zero,
            y = tf.divide((1. + (beta * beta)) * tp, ((1. + (beta * beta)) * tp )+ ((beta * beta) * fn) + fp)
        )

        sample_f0_5_score = fbeta_score(0.5, sample_precision, sample_recall)
        sample_f1_score = fbeta_score(1., sample_precision, sample_recall)
        sample_f2_score = fbeta_score(2., sample_precision, sample_recall)

        with tf.control_dependencies(metric_update_ops):
            micro_f1_score = fbeta_score(1., micro_precision, micro_recall)
            micro_f2_score = fbeta_score(2., micro_precision, micro_recall)
            micro_f0_5_score = fbeta_score(0.5, micro_precision, micro_recall)

        with tf.control_dependencies(metric_update_ops):
            macro_f1_score_class = []; macro_f2_score_class = []; macro_f0_5_score_class = []
            for tp_class, fp_class, tn_class, fn_class in zip(tp_class_accums, fp_class_accums, tn_class_accums, fn_class_accums):
                macro_f1_score_class.append(fbeta_score_core(1., tp_class, fp_class, tn_class, fn_class))
                macro_f2_score_class.append(fbeta_score_core(2., tp_class, fp_class, tn_class, fn_class))
                macro_f0_5_score_class.append(fbeta_score_core(0.5, tp_class, fp_class, tn_class, fn_class))
            macro_f1_score_class_avg = tf.reduce_mean(tf.stack(macro_f1_score_class))
            macro_f2_score_class_avg = tf.reduce_mean(tf.stack(macro_f2_score_class))
            macro_f0_5_score_class_avg = tf.reduce_mean(tf.stack(macro_f0_5_score_class))

        # Hamming Loss
        hamming_loss = tf.divide(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_xor(
                        tf.cast(predictions, tf.bool), 
                        tf.cast(labels, tf.bool)
                    ), 
                    tf.float64
                ),
                1
            ), 
            tf.cast(nb_class, tf.float64)
        )

        # Subset accuracy
        subset_accuracy = tf.cast(tf.reduce_all(tf.equal(predictions, labels), 1), tf.float64)

        # Jaccard index
        sample_accuracy = tf.where(tf.equal(tf.reduce_sum(label_union_prediction, 1), zero),
                    x = tf.zeros(dtype=tf.float64, shape=nb_ins),
                    y = tf.divide(tf.reduce_sum(true_positive, 1), tf.reduce_sum(label_union_prediction, 1)))

        with tf.control_dependencies(metric_update_ops):
            label_union_pred_class_sum = tf.add_n(label_union_pred_class_accums)
            micro_accuracy = tf.where(
                    tf.equal(label_union_pred_class_sum, zero), 
                    x = zero,
                    y = tf.divide(tf.add_n(tp_class_accums), label_union_pred_class_sum)
                )

        with tf.control_dependencies(metric_update_ops):
            macro_accuracy_class = []
            for tp_class, label_union_pred_class in zip(tp_class_accums, label_union_pred_class_accums):
                macro_accuracy_class.append(tf.divide(tp_class, label_union_pred_class))
            macro_accuracy_class_avg = tf.reduce_mean(tf.stack(macro_accuracy_class))

        # One-error
        row_indices = tf.range(nb_ins)
        col_indices = tf.argmax(probabilities, axis = 1, output_type=tf.int32)  
        one_error = tf.cast(tf.equal(tf.gather_nd(labels, tf.stack([row_indices, col_indices], axis = 1)), zero), tf.float64)

        # Coverage
        ranking = tf.reduce_sum(
            tf.cast(
                tf.less_equal(
                        tf.tile(tf.expand_dims(probabilities, -1), [1, 1, nb_class]),
                        tf.reshape(
                            tf.tile(
                                probabilities, 
                                [1, nb_class]
                            ), 
                        [nb_ins, nb_class, nb_class]
                        )
                    ),
                tf.float64
            ),
            2
        )
        coverage = tf.reduce_max(
            tf.where(
                tf.not_equal(labels, zero), 
                x = ranking, 
                y = tf.zeros(dtype=tf.float64, shape=[nb_ins, nb_class])
            ), 
            1
        )

        # Ranking Loss
        one_probs = tf.where(tf.not_equal(labels, zero), x = probabilities, y = tf.ones(dtype=tf.float64, shape=[nb_ins, nb_class]) * 2)
        zero_probs = tf.where(tf.equal(labels, zero), x = probabilities, y = tf.ones(dtype=tf.float64, shape=[nb_ins, nb_class]) * -1)
        ranking_loss = tf.where(
                            tf.equal(label_cardinality * tf.reduce_sum(tf.cast(tf.equal(labels, zero), tf.float64), 1), zero), 
                            x = tf.zeros(dtype=tf.float64, shape=nb_ins),
                            y = tf.divide(
                                    tf.reduce_sum(  
                                        tf.cast(
                                            tf.less_equal(
                                                    tf.tile(tf.expand_dims(one_probs, -1), [1, 1, nb_class]), 
                                                    tf.reshape(
                                                        tf.tile(zero_probs, [1, nb_class]), 
                                                        [nb_ins, nb_class, nb_class]
                                                    )
                                                ),
                                            tf.float64
                                        ), 
                                        [1,2]
                                    ), 
                                    label_cardinality * tf.reduce_sum(tf.cast(tf.equal(labels, zero), tf.float64), 1)
                                )
                            )
        # Label Ranking Average Precision
        one_ranking_lower = tf.where(
            tf.equal(labels, zero), 
            x = tf.ones(dtype=tf.float64, shape=[nb_ins, nb_class]) * 100, 
            y = ranking
        )

        one_ranking_greater = tf.where(
            tf.equal(labels, zero), 
            x = tf.zeros(dtype=tf.float64, shape=[nb_ins, nb_class]) * -1, 
            y = ranking
        )

        label_ranking_avg_precision = tf.where(
            tf.equal(tf.reduce_sum(labels,1), zero),
            x = tf.zeros(dtype=tf.float64, shape=nb_ins),
            y = tf.divide(
                tf.reduce_sum(
                    tf.divide(
                        tf.reduce_sum(
                            tf.cast(
                                tf.less_equal(
                                    tf.tile(tf.expand_dims(one_ranking_lower, -1), [1, 1, nb_class]),
                                    tf.reshape(
                                        tf.tile(one_ranking_greater, [1, nb_class]), 
                                        [nb_ins, nb_class, nb_class]
                                    )
                                ),
                                tf.float64
                            ),
                            1
                        ),
                        ranking
                    ),
                    1
                ),
                tf.reduce_sum(labels,1)
            )
        )

    with tf.variable_scope('statistics'):
        tf.summary.histogram('number_of_labels', label_cardinality)
        tf.summary.histogram('number_of_predictions', tf.reduce_sum(predictions, 1))
        tf.summary.histogram('probabilities', probabilities)

    metrics_list_to_average = [
            ('sample_precision', sample_precision), 
            ('sample_recall', sample_recall),  
            ('sample_f1_score', sample_f1_score), 
            ('sample_f2_score',sample_f2_score), 
            ('sample_f0_5_score', sample_f0_5_score), 
            ('hamming_loss', hamming_loss),
            ('subset_accuracy', subset_accuracy),
            ('sample_accuracy', sample_accuracy),
            ('one_error', one_error),
            ('coverage', coverage),
            ('label_cardinality', label_cardinality),
            ('ranking_loss', ranking_loss),
            ('label_ranking_avg_precision', label_ranking_avg_precision)
        ]

    metrics_list_to_accum = [
            ('micro_precision', micro_precision),  
            ('macro_precision_class_avg', macro_precision_class_avg), 
            ('micro_recall', micro_recall), 
            ('macro_recall_class_avg', macro_recall_class_avg), 
            ('micro_f1_score', micro_f1_score),
            ('micro_f2_score', micro_f2_score), 
            ('micro_f0_5_score', micro_f0_5_score), 
            ('micro_accuracy', micro_accuracy),
            ('macro_accuracy_class_avg', macro_accuracy_class_avg),
            ('macro_precision_class', macro_precision_class),
            ('macro_recall_class', macro_recall_class), 
            ('macro_f1_score_class', macro_f1_score_class),
            ('macro_f1_score_class_avg',macro_f1_score_class_avg),
            ('macro_f2_score_class', macro_f2_score_class), 
            ('macro_f2_score_class_avg', macro_f2_score_class_avg),
            ('macro_f0_5_score_class', macro_f0_5_score_class),
            ('macro_f0_5_score_class_avg',macro_f0_5_score_class_avg),
            ('macro_accuracy_class', macro_accuracy_class)
    ]

    metric_means = [[], []]
    metric_names = [[], []]
    for (name, var) in metrics_list_to_average:
        mean, update_op = tf.metrics.mean(var)
        metric_update_ops.append(update_op)
        metric_means[0].append(mean)
        metric_names[0].append(name)
        tf.summary.scalar(name, mean)

    for (name, metric_vars) in metrics_list_to_accum:
        if not type(metric_vars) == type([]):
            metric_names[1].append(name)
            metric_means[1].append(metric_vars)
            tf.summary.scalar(name, metric_vars)
        else:
            for j, var in enumerate(metric_vars):
                metric_names[1].append(name + str(j))
                metric_means[1].append(var)
                tf.summary.scalar(name + '/' + str(j), var)

    return metric_names, metric_means, metric_update_ops

def sparse_to_dense(indices, values):
    res = [[]]
    idx_pointer = 0
    val_pointer = 0
    for i in indices:
        if i[0] != idx_pointer:
            idx_pointer = i[0]
            res.append([])
        res[idx_pointer].append(values[val_pointer])
        val_pointer += 1
    return res
            