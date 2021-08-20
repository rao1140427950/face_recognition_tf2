"""
This script was modified from https://github.com/4uiiurz1/keras-arcface/blob/master/metrics.py
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss


class ArcFaceLoss(Loss):
    
    def __init__(self, s=30.0, m=0.50, **kwargs):
        super(ArcFaceLoss, self).__init__(**kwargs)
        self.s = s
        self.m = m

    def call(self, y_true, y_pred):
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(y_pred, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = y_pred * (1 - y_true) + target_logits * y_true
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits, axis=-1)
        # calc loss
        loss = tf.losses.categorical_crossentropy(y_true, out)
        loss = tf.reduce_mean(loss)

        return loss

    def get_config(self):
        config = {
            's': self.s,
            'm': self.m,
        }

        return super(ArcFaceLoss, self).get_config().update(config)