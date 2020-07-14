import numpy as np
import tensorflow as tf


class CustomMetrics(tf.keras.metrics.Metric):
    def __init__(self, nb_class, name="custom_metrics", **kwargs):
        super(CustomMetrics, self).__init__(name=name, **kwargs)
        self._prediction_threshold = 0.5
        self._zero = tf.constant(0.0, dtype=tf.float64)
        self._nb_class = nb_class

        self._class_tp = []
        self._class_fp = []
        self._class_tn = []
        self._class_fn = []

        for c in range(0, nb_class):
            self._class_tp.append(
                self.add_weight(
                    name="tp_{}".format(c), initializer="zeros", dtype=tf.float64
                )
            )
            self._class_fp.append(
                self.add_weight(
                    name="fp_{}".format(c), initializer="zeros", dtype=tf.float64
                )
            )
            self._class_tn.append(
                self.add_weight(
                    name="tn_{}".format(c), initializer="zeros", dtype=tf.float64
                )
            )
            self._class_fn.append(
                self.add_weight(
                    name="fn_{}".format(c), initializer="zeros", dtype=tf.float64
                )
            )

        # Precision
        nb_label = tf.add_n(self._class_tp) + tf.add_n(self._class_fp)
        self._micro_precision = tf.where(
            tf.equal(nb_label, self._zero),
            x=self._zero,
            y=tf.divide(tf.add_n(self._class_tp), nb_label),
        )

        macro_precision_class = []
        for tp_class, fp_class in zip(self._class_tp, self._class_fp):
            nb_predict_class = tp_class + fp_class
            macro_precision_class.append(
                tf.where(
                    tf.equal(nb_predict_class, self._zero),
                    x=self._zero,
                    y=tf.divide(tp_class, nb_predict_class),
                )
            )
        self._macro_precision_class_avg = tf.reduce_mean(
            tf.stack(macro_precision_class)
        )

        # Recall
        nb_label = tf.add_n(self._class_tp) + tf.add_n(self._class_fn)
        self._micro_recall = tf.where(
            tf.equal(nb_label, self._zero),
            x=self._zero,
            y=tf.divide(tf.add_n(self._class_tp), nb_label),
        )

        macro_recall_class = []
        for tp_class, fn_class in zip(self._class_tp, self._class_fn):
            nb_label_class = tp_class + fn_class
            macro_recall_class.append(
                tf.where(tf.equal(nb_label_class, self._zero), x = self._zero, y = tf.divide(tp_class, nb_label_class))
            )
        self._macro_recall_class_avg = tf.reduce_mean(tf.stack(macro_recall_class))


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float64)
        predictions = tf.cast(y_pred >= self._prediction_threshold, tf.float64)

        true_positive = tf.cast(
            tf.logical_and(
                tf.not_equal(predictions, self._zero), tf.not_equal(y_true, self._zero)
            ),
            tf.float64,
        )
        false_positive = tf.cast(
            tf.logical_and(
                tf.not_equal(predictions, self._zero), tf.equal(y_true, self._zero)
            ),
            tf.float64,
        )
        true_negative = tf.cast(
            tf.logical_and(
                tf.equal(predictions, self._zero), tf.equal(y_true, self._zero)
            ),
            tf.float64,
        )
        false_negative = tf.cast(
            tf.logical_and(
                tf.equal(predictions, self._zero), tf.not_equal(y_true, self._zero)
            ),
            tf.float64,
        )

        label_cardinality = tf.reduce_sum(y_true, 1)

        nb_ins = tf.shape(y_true)[0]
        nb_class = tf.shape(y_true)[1]

        for iclass, i in enumerate(tf.unstack(true_positive, axis=1)):
            self._class_tp[iclass].assign(tf.reduce_sum(i))

        for iclass, i in enumerate(tf.unstack(false_positive, axis=1)):
            self._class_fp[iclass].assign(tf.reduce_sum(i))

        for iclass, i in enumerate(tf.unstack(true_negative, axis=1)):
            self._class_tn[iclass].assign(tf.reduce_sum(i))

        for iclass, i in enumerate(tf.unstack(false_negative, axis=1)):
            self._class_fn[iclass].assign(tf.reduce_sum(i))

    def result(self):
        return (
            self._micro_precision,
            self._macro_precision_class_avg,
            self._micro_recall,
            self._macro_recall_class_avg,
        )

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        for c in range(0, self._nb_class):
            self._class_tp[c].assign(0.0)
            self._class_fp[c].assign(0.0)
            self._class_tn[c].assign(0.0)
            self._class_fn[c].assign(0.0)
