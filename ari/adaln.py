# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:21:44 2022

@author: AliaksandrSiarohin
"""
from keras.layers import Layer
from keras import backend as K

from keras.layers.core import Dense, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import UpSampling2D, Activation, add
from keras.layers.convolutional import Conv2D

def adain_block(inp, style, noise, fil, u = True):
    
    b = Dense(fil)(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil)(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
    
    if u:
        out = UpSampling2D(interpolation = 'bilinear')(inp)
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    else:
        out = Activation('linear')(inp)
    
    out = AdaInstanceNormalization()([out, b, g])
    #out = add([out, n])
    out = LeakyReLU(0.01)(out)
    
    b = Dense(fil)(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil)(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
    
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = AdaInstanceNormalization()([out, b, g])
    #out = add([out, n])
    out = LeakyReLU(0.01)(out)
    
    return out

#Input b and g should be 1x1xC
class AdaInstanceNormalization(Layer):
    def __init__(self, 
             axis=-1,
             momentum=0.99,
             epsilon=1e-3,
             center=True,
             scale=True,
             **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs) #inherit values from Layer class (keras)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
    
    
    def build(self, input_shape):
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
    
        super(AdaInstanceNormalization, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))
        
        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]