import sys
sys.path.append('..')
from utils.model import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
import tensorflow as tf
# from config import *


class MobileNetV2_(Model):

    def __init__(self,
                 input_shape=(224, 224, 3),
                 dropout=0.5,
                 num_classes=2,
                 regularizer=l2(0.0008),
                 alpha=1.0,
                 ):
        in_layer = Input(shape=input_shape, name='input_image')

        basemodel = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_shape=input_shape,
                                                           weights='imagenet', input_tensor=in_layer, alpha=alpha)

        x = GlobalAveragePooling2D()(basemodel.output)

        x = Dropout(dropout)(x)

        x = Dense(num_classes, name='features', kernel_regularizer=regularizer,
                  bias_regularizer=regularizer)(x)

        mod = tf.keras.models.Model(inputs=in_layer, outputs=x, name='resnet')
        super().__init__(basemodel_names=[], model=mod, kernel_regularizer=regularizer,
                         basemodel=basemodel)


if __name__ == '__main__':
    MobileNetV2_(
        input_shape=(112, 112, 3),
        num_classes=256,
    ).plot_model('mobilenet_v2.png')