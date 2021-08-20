import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy


class ArcFaceAccuracy(CategoricalAccuracy):
    
    def __init__(self, name='accuracy', **kwargs):
        super(ArcFaceAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        super(ArcFaceAccuracy, self).update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        return super(ArcFaceAccuracy, self).result()
