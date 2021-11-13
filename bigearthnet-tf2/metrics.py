import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from pprint import pprint

# TP/FP/TN/FN per class
# Precision
# Mean Precision per class
# Recall
# F1 Score

class CustomMetrics(tf.keras.metrics.Metric):
    def __init__(self, nb_class, name="custom_metrics", **kwargs):
        super(CustomMetrics, self).__init__(name=name, **kwargs)
        self._prediction_threshold = 0.5
        self._zero = tf.constant(0.0, dtype=tf.float64)
        self._nb_class = nb_class

        self._class_tp = self.add_weight(
            name="tp", shape=(nb_class,), initializer="zeros", dtype=tf.float64
        )
        self._class_fp = self.add_weight(
            name="fp", shape=(nb_class,), initializer="zeros", dtype=tf.float64
        )
        self._class_tn = self.add_weight(
            name="tn", shape=(nb_class,), initializer="zeros", dtype=tf.float64
        )
        self._class_fn = self.add_weight(
            name="fn", shape=(nb_class,), initializer="zeros", dtype=tf.float64
        )
        self._class_label_union_pred = self.add_weight(
            name="label_union_pred",
            shape=(nb_class,),
            initializer="zeros",
            dtype=tf.float64,
        )

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
        label_union_prediction = tf.cast(
            tf.logical_or(tf.not_equal(predictions, self._zero), tf.not_equal(y_true, self._zero)),
            tf.float64,
        )



        self._class_tp.assign_add(tf.reduce_sum(true_positive, axis=0))
        self._class_fp.assign_add(tf.reduce_sum(false_positive, axis=0))
        self._class_tn.assign_add(tf.reduce_sum(true_negative, axis=0))
        self._class_fn.assign_add(tf.reduce_sum(false_negative, axis=0))
        self._class_label_union_pred.assign_add(tf.reduce_sum(label_union_prediction, axis=0))

        flat_confusion_matrix = tf.convert_to_tensor(
            [
                self._class_tn,
                self._class_fp,
                self._class_fn,
                self._class_tp,
            ]
        )
        self._confusion_matrix = tf.reshape(tf.transpose(flat_confusion_matrix), [-1, 2, 2])


    def result(self):
        # Precision
        nb_label = tf.reduce_sum(self._class_tp) + tf.reduce_sum(self._class_fp)
        micro_precision = tf.where(
            tf.equal(nb_label, self._zero),
            x=self._zero,
            y=tf.divide(tf.reduce_sum(self._class_tp), nb_label),
        )

        nb_predict_class = self._class_tp + self._class_fp

        macro_precision_class = tf.where(
            tf.equal(nb_predict_class, self._zero),
            x=self._zero,
            y=tf.divide(self._class_tp, nb_predict_class),
        )
        macro_precision = tf.reduce_mean(macro_precision_class)

        # Recall
        nb_label = tf.reduce_sum(self._class_tp) + tf.reduce_sum(self._class_fn)
        micro_recall = tf.where(
            tf.equal(nb_label, self._zero),
            x=self._zero,
            y=tf.divide(tf.reduce_sum(self._class_tp), nb_label),
        )

        nb_label_class = self._class_tp + self._class_fn
        macro_recall_class = tf.where(
            tf.equal(nb_label_class, self._zero),
            x=self._zero,
            y=tf.divide(self._class_tp, nb_label_class),
        )
        macro_recall = tf.reduce_mean(macro_recall_class)

        # Accuracy
        label_union_pred_class_sum = tf.reduce_sum(self._class_label_union_pred)
        micro_accuracy = tf.where(
            tf.equal(label_union_pred_class_sum, self._zero),
            x = self._zero,
            y = tf.divide(tf.reduce_sum(self._class_tp), label_union_pred_class_sum)
        )

        label_union_pred_class = self._class_label_union_pred
        macro_accuracy_class = tf.where(
            tf.equal(label_union_pred_class, self._zero),
            x=self._zero,
            y=tf.divide(self._class_tp, label_union_pred_class),
        )
        macro_accuracy = tf.reduce_mean(macro_accuracy_class)

        #F1-Score
        micro_fscore = tf.where(
            tf.logical_and(
                tf.equal(micro_precision, self._zero), tf.equal(micro_recall, self._zero))
                           , x=self._zero, y= 2*(micro_precision*micro_recall)/(micro_precision + micro_recall))
        
        macro_fscore_class = tf.where(
            tf.logical_and(
                tf.equal(macro_precision_class, self._zero), tf.equal(macro_recall_class, self._zero))
                           , x=self._zero, y= 2*(macro_precision_class*macro_recall_class)/(macro_precision_class + macro_recall_class))
        macro_fscore = tf.reduce_mean(macro_fscore_class)

        # pprint(dict(zip([
        #                 "Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops", "Pastures",
        #                 "Complex cultivation patterns", "Land principally occupied by agriculture, with significant areas of natural vegetation",
        #                 "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland and sparsely vegetated areas",
        #                 "Moors, heathland and sclerophyllous vegetation", "Transitional woodland, shrub", "Beaches, dunes, sands", "Inland wetlands", 
        #                 "Coastal wetlands", "Inland waters", "Marine waters"
        #                 ], np.round(macro_fscore_class.numpy(),4))))

        return (
            micro_precision,
            macro_precision,
            micro_recall,
            macro_recall,
            micro_accuracy,
            macro_accuracy,
            micro_fscore,
            macro_fscore
        )

    def reset_states(self):
        K.batch_set_value([(v, np.zeros((self._nb_class,))) for v in self.variables])
