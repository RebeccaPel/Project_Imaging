# -*- coding: utf-8 -*-
"""
Definition of custom classes based on keras.Layer for several purposes:
    - Minibatch Discriminator - based on Salimans et. al. (2016)
    - ResBlock
    - ResBlockUp
    - ResBlockDown
    - UpSampleBlock
    
@author: N. Hartog, J. Kleinveld, A. Masot, R. Pelsser
"""

import keras
from keras import layers
from keras.layers import Conv2D, UpSampling2D, LeakyReLU, BatchNormalization
from keras import backend as K
from keras.layers import Layer
from keras.layers import InputSpec
from keras import initializers, regularizers, constraints
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# code source: https://github.com/forcecore/Keras-GAN-Animeface-Character/blob/master/discrimination.py
class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.
    # Example
    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)
        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```
    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.
    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
                                 initializer=self.init,
                                 name='kernel',
                                 regularizer=self.W_regularizer,
                                 trainable=True,
                                 constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1] + self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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


class UpSampleBlock(Layer):
    """
    Convolves, upsamples and applies leaky relu activation to the input.
    Inherits from the keras.layers.Layer class
    # Arguments:
    dimension: number of filters used in the convolutional layer.
    kernel_size: size of the convolutional kernel
    """

    def __init__(self, dimension, kernel_size=(3, 3), **kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        self.dimension = dimension
        self.kernel_size = kernel_size
        # Conv2Dtranspose let to checkerboarding, this could be fixed by changing kernel size.
        # But this is not desirable.
        self.conv2d = Conv2D(dimension, kernel_size=kernel_size, padding='same')
        self.upsample = UpSampling2D(size=(2, 2))
        self.leakyrelu = LeakyReLU(0.2)
        self.bn = BatchNormalization(momentum=0.3)

    def get_config(self):
        config = super(UpSampleBlock, self).get_config()
        config.update({
            'dimension': self.dimension,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, input):
        x = self.upsample(input)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x