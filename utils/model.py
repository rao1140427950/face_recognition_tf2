import abc
import os
import numpy as np
import tensorflow as tf


class Model(metaclass=abc.ABCMeta):

    def __init__(self, basemodel_names=None, model=None, basemodel=None, kernel_regularizer=None, r=4):
        if basemodel_names is None:
            basemodel_names = []
        self.model = model
        self.kernel_regularizer = kernel_regularizer
        self.basemodel_names = basemodel_names
        self.basemodel = basemodel
        self.r = r

    def __call__(self, x):
        return self.model(x)

    def pool(self, inputs, method='max', pool_size=3, strides=1, padding='same', name=None):
        if method == 'max':
            return self.maxpool_layer(inputs, pool_size=pool_size, strides=strides, padding=padding, name=name)
        elif method == 'avg':
            return self.avgpool_layer(inputs, pool_size=pool_size, strides=strides, padding=padding, name=name)
        else:
            raise ValueError('Pooling method should be `avg` or `max` but get `{}`.'.format(method))

    def conv_layer(self, inputs, filters, kernel_size, strides=1, padding='same', activation='relu', name=None, bn=True):
        conv_name = None
        bn_name = None
        act_name = None
        if name is not None:
            conv_name = name + '_conv'
            bn_name = name + '_bn'
            act_name = name + '_' + activation
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            use_bias=False,
            kernel_regularizer=self.kernel_regularizer,
            name=conv_name
        )(inputs)
        if bn:
            x = tf.keras.layers.BatchNormalization(scale=False, name=bn_name)(x)
        if activation:
            x = tf.keras.layers.Activation(activation, name=act_name)(x)
        return x

    @staticmethod
    def maxpool_layer(inputs, pool_size=3, strides=1, padding='same', name=None):
        return tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)(inputs)

    @staticmethod
    def avgpool_layer(inputs, pool_size=3, strides=1, padding='same', name=None):
        return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)(inputs)

    def freeze_layers(self, layer_name):
        check = False
        for layer in self.model.layers:
            if check:
                layer.trainable = True
            else:
                layer.trainable = False
            if layer.name == layer_name:
                check = True
        return

    def freeze_base_layers(self):
        if self.basemodel is not None:
            for layer in self.basemodel.layers:
                layer.trainable = False
            print('Freeze base layers by basemodel.')
        else:
            for layer in self.model.layers:
                if layer.name in self.basemodel_names:
                    layer.trainable = False
                else:
                    layer.trainable = True
            print('Freeze base layers by basemodel_names.')
        return

    def release_all_layers(self):
        for layer in self.model.layers:
            layer.trainable = True
        return

    def summary(self):
        for layer in self.model.layers:
            print(layer.name, layer.trainable)
        return

    def load_weights(self, weights_file, **kwargs):
        self.model.load_weights(weights_file, **kwargs)
        return

    def save_weights(self, weights_file, **kwargs):
        self.model.save_weights(weights_file, **kwargs)
        return

    def plot_model(self, filename):
        tf.keras.utils.plot_model(
            self.model,
            to_file = filename,
            show_shapes=True,
            show_layer_names=True
        )
        return

    def print_json(self):
        print(self.model.to_json())
        return

    def save_json(self, filename):
        with open(filename, 'w') as f:
            f.write(self.model.to_json())
            f.close()
        return

    def print_layer_names(self):
        names = []
        for layer in self.model.layers:
            names.append(layer.name)
        print(names)
        return

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        return

