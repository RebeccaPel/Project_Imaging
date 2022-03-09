# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:24:35 2022

@author: Ari
"""
from keras.layers import add, Conv2D, Activation

## ResNet layer

def residual_module(layer_in, n_filters):
    """ 
    Adds a ResNet conv2d 3x3 layer with stride 1 and padding same.

    Args:
        layer_in: The previous layer of the model. If it's the first one, use an Input() layer.
        n_filters: Number of filters of the two identical convolutions performed.

    Returns:
        The output layer.
    """
    merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
    conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
    conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
    layer_out = add([conv2, merge_input])
	# activation function
    layer_out = Activation('relu')(layer_out)
    return layer_out
