# from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
# from tensorflow.keras import regularizers

import tensorflow as tf


class ArcFaceLayer(Layer):

    def __init__(self, n_classes=10, regularizer=None, **kwargs):
        super(ArcFaceLayer, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.regularizer = regularizer
        self.w = None

    def build(self, input_shape):
        super(ArcFaceLayer, self).build(input_shape)
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1], self.n_classes),
            initializer='glorot_uniform',
            trainable=True,
            regularizer=self.regularizer,
        )

    def call(self, inputs, **kwargs):
        x = tf.nn.l2_normalize(inputs, axis=-1)
        w = tf.nn.l2_normalize(self.w, axis=0)
        cos = tf.matmul(x, w)

        return cos

    def get_config(self):
        config = {
            'n_classes': self.n_classes
        }

        return super(ArcFaceLayer, self).get_config().update(config)