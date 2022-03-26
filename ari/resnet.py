# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:24:35 2022

@author: Ari
"""
import keras
from keras.layers import Conv2D

import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv2a = Conv2D(filters[0], kernel_size, padding='same',
                             kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2a = keras.layers.BatchNormalization()

        self.conv2b = Conv2D(filters[1], kernel_size, padding='same',
                             kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2b = keras.layers.BatchNormalization()
        self.av2b = keras.layers.AvgPool2D((2, 2))

        self.conv2in = Conv2D(filters[1], 1, padding='same',
                              kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))

    def get_config(self):
        config = super(ResBlock, self).get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, input_tensor, training=True):
        x = self.bn2a(input_tensor, training=training)
        x = tf.nn.relu(x)
        x = self.conv2a(x)

        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        x += input_tensor
        return x


class ResBlockDown(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResBlockDown, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        self.conv2a = Conv2D(filters[0], (1, 1), padding='same',
                             kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2a = keras.layers.BatchNormalization()

        self.conv2b = Conv2D(filters[1], kernel_size, padding='same', strides=(2, 2),
                             kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2b = keras.layers.BatchNormalization()
        # self.av2b = keras.layers.AvgPool2D((2, 2))

        self.conv2in = Conv2D(filters[1], 1, padding='same', strides=(2, 2),
                              kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        # self.av2in = keras.layers.AvgPool2D((2, 2))

    def get_config(self):
        config = super(ResBlockDown, self).get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, input_tensor):
        x = self.conv2a(input_tensor)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = tf.nn.relu(x)
        # x = self.av2b(x)

        skip = self.conv2in(input_tensor)
        # skip = self.av2in(skip)
        x += skip
        return x


class ResBlockUp(keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), **kwargs):
        super(ResBlockUp, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.up2a = keras.layers.UpSampling2D((2, 2))

        self.conv2a = Conv2DTranspose(filters[0], kernel_size, padding='same', strides=(2, 2),
                                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2a = keras.layers.BatchNormalization()

        self.conv2b = Conv2D(filters[1], kernel_size, padding='same',
                             kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2b = keras.layers.BatchNormalization()

        self.conv2in = Conv2DTranspose(filters[1], 1, padding='same', strides=(2, 2),
                                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.up2in = keras.layers.UpSampling2D((2, 2))

    def get_config(self):
        config = super(ResBlockUp, self).get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, input_tensor, training=True):
        x = self.bn2a(input_tensor, training=training)
        x = tf.nn.relu(x)
        # x = self.up2a(x)
        x = self.conv2a(x)

        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        # skip = self.up2in(input_tensor)
        skip = self.conv2in(input_tensor)
        x += skip
        return x