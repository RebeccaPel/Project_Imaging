# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:24:35 2022

@author: Ari
"""
import keras
from keras.layers import Conv2D

import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal


class ResBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), *kwargs):
        super().__init__()
        self.conv2a = Conv2D(filters[0], kernel_size, padding='same', kernel_initializer=Orthogonal)
        self.bn2a = keras.layers.BatchNormalization()

        self.conv2b = Conv2D(filters[1], kernel_size, padding='same', kernel_initializer=Orthogonal)
        self.bn2b = keras.layers.BatchNormalization()
        self.av2b = keras.layers.AvgPool2D((2, 2))

        self.conv2in = Conv2D(filters[1], 1, padding='same')

    def call(self, input_tensor, training=True):
        x = self.bn2a(input_tensor, training=training)
        x = tf.nn.relu(x)
        x = self.conv2a(x)

        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        x += input_tensor
        return x


class ResBlockDown(tf.keras.Model):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__()
        self.conv2a = Conv2D(filters[0], (1, 1), padding='same')
        self.bn2a = keras.layers.BatchNormalization()

        self.conv2b = Conv2D(filters[1], kernel_size, padding='same')
        self.bn2b = keras.layers.BatchNormalization()
        self.av2b = keras.layers.AvgPool2D((2, 2))

        self.conv2in = Conv2D(filters[1], 1, padding='same')
        self.av2in = keras.layers.AvgPool2D((2, 2))

    def call(self, input_tensor):
        x = tf.nn.relu(input_tensor)
        x = self.conv2a(x)

        x = tf.nn.relu(x)
        x = self.conv2b(x)
        x = self.av2b(x)

        skip = self.conv2in(input_tensor)
        skip = self.av2in(skip)
        x += skip
        return x


class ResBlockUp(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), **kwargs):
        super().__init__()
        self.up2a = keras.layers.UpSampling2D((2, 2))
        self.conv2a = Conv2D(filters[0], kernel_size, padding='same')
        self.bn2a = keras.layers.BatchNormalization()

        self.conv2b = Conv2D(filters[1], kernel_size, padding='same')
        self.bn2b = keras.layers.BatchNormalization()
        self.av2b = keras.layers.AvgPool2D((2, 2))

        self.conv2in = Conv2D(filters[1], 1, padding='same')
        self.up2in = keras.layers.UpSampling2D((2, 2))

    def call(self, input_tensor, training=True):
        x = self.bn2a(input_tensor, training=training)
        x = tf.nn.relu(x)
        x = self.up2a(x)
        x = self.conv2a(x)

        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        skip = self.up2in(input_tensor)
        skip = self.conv2in(skip)
        x += skip
        return x