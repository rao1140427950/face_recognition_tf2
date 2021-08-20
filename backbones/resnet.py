import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Add, Dropout
import tensorflow.keras.backend as K
import sys
sys.path.append('..')
from utils.model import Model
# from config import *


class ResNet_(Model):

    def __init__(self,
                 input_shape=(224, 224, 3),
                 # fc_size=256,
                 dropout=0.5,
                 num_classes=2,
                 kernel_regularizer=l2(0.0008),
                 repetitions=(3, 4, 6, 3)
                 ):
        in_layer = Input(shape=input_shape, name='input_image')

        if repetitions == (3, 4, 6, 3):
            basemodel = tf.keras.applications.resnet.ResNet50(include_top=False, input_shape=input_shape,
                                                              weights='imagenet', input_tensor=in_layer)
        elif repetitions == (3, 4, 23, 3):
            basemodel = tf.keras.applications.resnet.ResNet101(include_top=False, input_shape=input_shape,
                                                               weights='imagenet', input_tensor=in_layer)
        elif repetitions == (3, 8, 36, 3):
            basemodel = tf.keras.applications.resnet.ResNet152(include_top=False, input_shape=input_shape,
                                                               weights='imagenet', input_tensor=in_layer)
        else:
            raise ValueError('`repetitions is not supported.')

        x = GlobalAveragePooling2D()(basemodel.output)

        x = Dropout(dropout)(x)

        x = Dense(num_classes, name='features', kernel_regularizer=kernel_regularizer,
                  bias_regularizer=kernel_regularizer)(x)

        mod = tf.keras.models.Model(inputs=in_layer, outputs=x, name='resnet')
        super().__init__(basemodel_names=[], model=mod, kernel_regularizer=kernel_regularizer,
                         basemodel=basemodel)


class ResNet(Model):

    def _build_model(self):
        in_layer = Input(shape=self.input_shape)

        x = ZeroPadding2D((3, 3), name='conv1_pad')(in_layer)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1_conv', kernel_initializer='he_normal',
                   kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
        x = BatchNormalization(axis=self.bn_axis, name='conv1_bn')(x)
        x = Activation(self.activation, name='conv1_relu')(x)
        x = ZeroPadding2D((1, 1), name='pool1_pad')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_pool')(x)

        x = self.conv_block(x, 3, [64, 64, 256], block='conv2_block1', strides=(1, 1))
        for r in range(self.repetitions[0] - 1):
            x = self.identity_block(x, 3, [64, 64, 256], block='conv2_block' + str(r + 2))

        x = self.conv_block(x, 3, [128, 128, 512], block='conv3_block1')
        for r in range(self.repetitions[1] - 1):
            x = self.identity_block(x, 3, [128, 128, 512], block='conv3_block' + str(r + 2))

        x = self.conv_block(x, 3, [256, 256, 1024], block='conv4_block1')
        for r in range(self.repetitions[2] - 1):
            x = self.identity_block(x, 3, [256, 256, 1024], block='conv4_block' + str(r + 2))

        x = self.conv_block(x, 3, [512, 512, 2048], block='conv5_block1')
        for r in range(self.repetitions[3] - 1):
            x = self.identity_block(x, 3, [512, 512, 2048], block='conv5_block' + str(r + 2))

        x = GlobalAveragePooling2D()(x)

        # FC Block
        x = Dropout(self.dropout)(x)
        x = Dense(self.fc_size, activation=self.activation, name='fc' + str(self.fc_size),
                  kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
        x = Dropout(self.dropout)(x)
        x = Dense(self.num_classes, activation='softmax', name='predictions',
                  kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)

        mod = tf.keras.models.Model(inputs=in_layer, outputs=x, name='resnet')

        return mod

    def identity_block(self, input_tensor, kernel_size, filters, block):

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = Conv2D(filters1, (1, 1), name=block + '_1_conv', kernel_initializer='he_normal',
                   kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=block + '_1_bn')(x)
        x = Activation(self.activation, name=block + '_1_relu')(x)

        x = Conv2D(filters2, kernel_size, kernel_initializer='he_normal', padding='same', name=block + '_2_conv',
                   kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
        x = BatchNormalization(axis=bn_axis, name=block + '_2_bn')(x)
        x = Activation(self.activation, name=block + '_2_relu')(x)

        x = Conv2D(filters3, (1, 1), name=block + '_3_conv', kernel_initializer='he_normal',
                   kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
        x = BatchNormalization(axis=bn_axis, name=block + '_3_bn')(x)

        x = Add(name=block + '_add')([x, input_tensor])
        x = Activation(self.activation, name=block + '_out')(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, block, strides=(2, 2)):

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', name=block + '_1_conv',
                   kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=block + '_1_bn')(x)
        x = Activation(self.activation, name=block + '_1_relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=block + '_2_conv',
                   kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
        x = BatchNormalization(axis=bn_axis, name=block + '_2_bn')(x)
        x = Activation(self.activation, name=block + '_2_relu')(x)

        x = Conv2D(filters3, (1, 1), name=block + '_3_conv', kernel_initializer='he_normal',
                   kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
        x = BatchNormalization(axis=bn_axis, name=block + '_3_bn')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', name=block + '_0_conv',
                          kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=block + '_0_bn')(shortcut)

        x = Add(name=block + '_add')([x, shortcut])
        x = Activation(self.activation, name=block + '_out')(x)
        return x


    def __init__(self,
                 input_shape=(224, 224, 3),
                 fc_size=256,
                 num_classes=2,
                 dropout=0.5,
                 regularizer=l2(0.0008),
                 repetitions=(3, 4, 23, 3),
                 activation='relu',
                 ):

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        self.input_shape = input_shape
        self.fc_size = fc_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.repetitions = repetitions
        self.bn_axis = bn_axis
        self.regularizer = regularizer
        self.activation = activation

        mod = self._build_model()

        super().__init__(basemodel_names=[], model=mod, kernel_regularizer=regularizer)
