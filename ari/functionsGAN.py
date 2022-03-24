# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:26:21 2022

@author: Ari
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from numpy.random import randn

import keras
keras.backend.set_image_data_format('channels_first')
from keras.models import Model
from keras.layers.core import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, LayerNormalization, MultiHeadAttention
from keras.layers.convolutional import Conv2D, Conv2DTranspose

#import adaln
from resnet import ResBlock


def mapping(noise):
    layer_in = ResBlock([200,200])(noise)
    layer_in = Dense(200)(layer_in)
    layer_in = LeakyReLU(0.2)(layer_in)
    layer_in = ResBlock([200,200])(layer_in)
    layer_in = Dense(200)(layer_in)
    layer_in = LeakyReLU(0.2)(layer_in)
    layer_in = ResBlock([200,200])(layer_in)
    layer_in = Dense(200)(layer_in)
    layer_in = LeakyReLU(0.2)(layer_in)
    layer_in = ResBlock([200,200])(layer_in)
    layer_in = Dense(200)(layer_in)
    layer_in = LeakyReLU(0.2)(layer_in)
    w = Dense(200)(layer_in)
    return w
 

def get_discriminator_histopathology():
   """
   The histopathology discriminator, based on the architecture of the Pathology GAN
   
   Args:
       input_size: The size of the images, as an int (eg. 28 means a 28x28 RGB image is used).
       
   Returns:
       The discriminator model.
   """
   #image_input = Input(shape=(3,224, 224))
   image_input = Input(shape=(3,96, 96))

   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 3
   print(image_input.shape)
   layer_in = ResBlock([3,3])(image_input)
   print('d_resnet1',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu1',layer_in.shape)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2, 32
   layer_in = Conv2D(32, kernel_size=(2, 2), strides=(2, 2))(layer_in)
   print('d_conv1',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu2',layer_in.shape)
   
   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 32
   layer_in = ResBlock([32,32])(layer_in)
   print('d_resnet2',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu3',layer_in.shape)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2, 64
   layer_in = Conv2D(64, kernel_size=(2, 2), strides=(2, 2))(layer_in)
   print('d_conv2',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu4',layer_in.shape)
   
   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 64
   layer_in = ResBlock([64,64])(layer_in)
   print('d_resnet3',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu5',layer_in.shape)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2, 128
   layer_in = Conv2D(128, kernel_size=(2, 2), strides=(2, 2))(layer_in)
   print('d_conv3',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu6',layer_in.shape)
   
   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 128
   layer_in = ResBlock([128,128])(layer_in)
   print('d_resnet4',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu7',layer_in.shape)
   
   #Attention Layer at 28x28x128 - No parameters are specified, nor the type of attention layer, so default for now (Dot product)
   layer_in = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))(layer_in,layer_in)
   print('d_attention',layer_in.shape)
   #layer_in = Attention()(layer_in)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2, 256 #took the padding as none
   layer_in = Conv2D(256, kernel_size=(2, 2), strides=(2, 2))(layer_in)
   print('d_conv4',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu8',layer_in.shape)
   
   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 256
   layer_in = ResBlock([256,256])(layer_in)
   print('d_resnet5',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu9',layer_in.shape)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2, 512
   layer_in = Conv2D(512, kernel_size=(2, 2), strides=(2, 2))(layer_in)
   print('d_conv5',layer_in.shape)
   layer_in = LeakyReLU(0.2)(layer_in)
   print('d_relu10',layer_in.shape)
   
   layer_in = Flatten()(layer_in) #7x7x512
   print('d_flatten',layer_in.shape)
   layer_in = Dense(1024)(layer_in)
   print('d_dense1',layer_in.shape)
   layer_in = LeakyReLU()(layer_in)
   print('d_relu11',layer_in.shape)
   layer_out = Dense(1, activation='leaky_relu')(layer_in)
   print('d_dense2',layer_out.shape)
   
   discriminator = Model(inputs = image_input, outputs = layer_out)
   
   return discriminator
   

def get_generator_histopathology(latent_dim = 200): 
    """
    The histopathology generator, based on the architecture of the Pathology GAN
    
    Args:
        latent_dim: The latent dimension
        
    Returns:
        The generator model.
    """
    inputs = Input(shape = (latent_dim,))
    layer_in = Dense(1024, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    print('Dense1',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('Norm1',layer_in.shape)
    layer_in = LeakyReLU()(layer_in) #Ran with default becasue 0.2 is not specified, default is 0.3
    print('ReLu1',layer_in.shape)
    
    layer_in = Dense(2304, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(layer_in)
    print('Dense2',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('Norm2',layer_in.shape)
    layer_in = LeakyReLU()(layer_in) #Ran with default becasue 0.2 is not specified, default is 0.3
    print('ReLu2',layer_in.shape)
    
    layer_in = Reshape((256, 3, 3))(layer_in)
    print('Reshape1',layer_in.shape)
    
    ##1
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 256
    layer_in = ResBlock([256,256])(layer_in)
    print('resnet1',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm3',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu3',layer_in.shape)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2, 512
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling1',layer_in.shape)
    layer_in = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in) #CHECK IF UPSCALE WORKS
    print('conv1',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm4',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu4',layer_in.shape)
    
    ##2
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 512
    layer_in = ResBlock([512,512])(layer_in)
    print('resnet2',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm5',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu5',layer_in.shape)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2, 256
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling2',layer_in.shape)
    layer_in = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in) #CHECK IF UPSCALE WORKS
    print('conv2',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm6',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu6',layer_in.shape)
    
    ##3 
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 256
    layer_in = ResBlock([256,256])(layer_in)
    print('resnet3',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm7',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu7',layer_in.shape)
    
    #Attention Layer at 28x28x256 - No parameters are specified, nor the type of attention layer, so default for now
    layer_in = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))(layer_in,layer_in)
    print('attention',layer_in.shape)
    #layer_in = Attention()(layer_in)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2, 128
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling3',layer_in.shape)
    layer_in = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in) #CHECK IF UPSCALE WORKS
    print('conv3',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm8',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu8',layer_in.shape)
    
    ##4
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 128
    layer_in = ResBlock([128,128])(layer_in)
    print('resnet4',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm9',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu9',layer_in.shape)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2, 64
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling4',layer_in.shape)
    layer_in = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in)
    print('conv4',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm10',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu10',layer_in.shape)
    
    ##5
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 64
    layer_in = ResBlock([64,64])(layer_in)
    print('resnet5',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm11',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu11',layer_in.shape)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2,32
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling5',layer_in.shape)
    layer_in = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in)
    print('conv5',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm12',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu12',layer_in.shape)
    
    layer_out = Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same', activation='tanh')(layer_in)
    print('conv6',layer_out.shape)
    generator = Model(inputs = inputs, outputs = layer_out)
    
    return generator

def combined_model(image_size, latent_dim):
     """
     The combined model of the histopathology generator and discriminator
     
     Args:
         latent_dim: The latent dimension.
         
     Returns:
         The combined model, the generator and the discriminator.
     """
     g = get_generator_histopathology()
     d = get_discriminator_histopathology()
     noise = Input(shape=(latent_dim,))
     g_out = g(noise)
     d.trainable = False
     d_out = d(g_out)
     model = keras.models.Model(inputs=noise, outputs=d_out)
     model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
     return model, g, d
    """
    The histopathology generator, based on the architecture of the Pathology GAN
    
    Args:
        latent_dim: The latent dimension
        
    Returns:
        The generaot model.
    """
    inputs = Input(shape = (latent_dim,))
    layer_in = Dense(1024, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    print('Dense1',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('Norm1',layer_in.shape)
    layer_in = LeakyReLU()(layer_in) #Ran with default becasue 0.2 is not specified, default is 0.3
    print('ReLu1',layer_in.shape)
    
    layer_in = Dense(12544, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(layer_in)
    print('Dense2',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('Norm2',layer_in.shape)
    layer_in = LeakyReLU()(layer_in) #Ran with default becasue 0.2 is not specified, default is 0.3
    print('ReLu2',layer_in.shape)
    
    layer_in = Reshape((256, 7, 7))(layer_in)
    print('Reshape1',layer_in.shape)
    
    ##1
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 256
    layer_in = residual_module(layer_in, 256)
    print('resnet1',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm3',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu3',layer_in.shape)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2, 512
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling1',layer_in.shape)
    layer_in = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in) #CHECK IF UPSCALE WORKS
    print('conv1',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm4',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu4',layer_in.shape)
    
    ##2
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 512
    layer_in = residual_module(layer_in, 512)
    print('resnet2',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm5',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu5',layer_in.shape)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2, 256
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling2',layer_in.shape)
    layer_in = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in) #CHECK IF UPSCALE WORKS
    print('conv2',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm6',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu6',layer_in.shape)
    
    ##3 
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 256
    layer_in = residual_module(layer_in, 256)
    print('resnet3',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm7',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu7',layer_in.shape)
    
    #Attention Layer at 28x28x256 - No parameters are specified, nor the type of attention layer, so default for now
    layer_in = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))(layer_in,layer_in)
    print('attention',layer_in.shape)
    #layer_in = Attention()(layer_in)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2, 128
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling3',layer_in.shape)
    layer_in = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in) #CHECK IF UPSCALE WORKS
    print('conv3',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm8',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu8',layer_in.shape)
    
    ##4
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 128
    layer_in = residual_module(layer_in, 128)
    print('resnet4',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm9',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu9',layer_in.shape)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2, 64
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling4',layer_in.shape)
    layer_in = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in) #CHECK IF UPSCALE WORKS
    print('conv4',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm10',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu10',layer_in.shape)
    
    ##5
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 64
    layer_in = residual_module(layer_in, 64)
    print('resnet5',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm11',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu11',layer_in.shape)
    
    #ConvTranspose2D Layer, 2x2, stride 2, pad upscale, AdaIN, and leakyReLU 0.2,32
    #layer_in = UpSampling2D()(layer_in)
    #print('upsampling5',layer_in.shape)
    layer_in = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2,2), padding='same')(layer_in) #CHECK IF UPSCALE WORKS
    print('conv5',layer_in.shape)
    layer_in = LayerNormalization()(layer_in)
    print('norm12',layer_in.shape)
    layer_in = LeakyReLU(0.2)(layer_in)
    print('relu12',layer_in.shape)
    
    layer_out = Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same', activation='tanh')(layer_in)
    print('conv6',layer_out.shape)
    generator = Model(inputs = inputs, outputs = layer_out)
    
    return generator


# def get_generator_histopathology_adain(image_size, latent_dim = 300): 
#     """
#     The histopathology generator, based on the architecture of the Pathology GAN
    
#     Args:
#         latent_dim: The latent dimension
        
#     Returns:
#         The generaot model.
#     """
#     inputs = Input(shape = (latent_dim,))
#     layer_in = Dense(1024, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
#     layer_in = Reshape([1024,1,1])(layer_in)
#     #AdaIN
#     #generate the style, beta and gamma
#     sty = Dense(12544, kernel_initializer = 'he_normal')(Input(shape = [latent_dim]))
#     sty = LeakyReLU(0.1)(sty)
#     sty = Dense(12544, kernel_initializer = 'he_normal')(sty)
#     sty = LeakyReLU(0.1)(sty)

#     layer_in =adaln.adain_block(layer_in, sty, 1024, u=False)
#     layer_in = LeakyReLU()(layer_in) #Ran with default becasue 0.2 is not specified, default is 0.3
    
#     layer_in = Dense(12544, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(layer_in)
#     #layer_in = Reshape([12544,1,1])(layer_in)
#     #AdaIN
#     layer_in =adaln.adain_block(layer_in, sty, 12544, 1)
#     layer_in = LeakyReLU()(layer_in) #Ran with default becasue 0.2 is not specified, default is 0.3
    
#     layer_in = Reshape((256, 7, 7))(layer_in)
    
#     ##1 resnet, adain, leakyrelu 0.2
#     #ResNet Conv2D Layer, 3x3, stride 1, pad same, 256
#     layer_in = residual_module(layer_in, 256)
#     #AdaIN
#     layer_in =adaln.adain_block(layer_in, sty, 256, 7)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     layer_in = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
#     #AdaIN
#     layer_in =adaln.adain_block(layer_in, sty, 256)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     ##2 resnet, adain, leakyrelu 0.2
#     #add ResNet Conv2D Layer, 3x3, stride 1, pad same, 512
#     layer_in = residual_module(layer_in, 512)
#     #AdaIN
#     layer_in =adaln.adain_block(layer_in, sty, 512)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     layer_in = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
#     #AdaIN
#     layer_in =adaln.adain_block(layer_in, sty, 256)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     ##3 resnet, attention layer, adain, leakyrelu 0.2
#     #add ResNet Conv2D Layer, 3x3, stride 1, pad same, 256
#     layer_in = residual_module(layer_in, 256)
#     #AdaIN
#     layer_in =adaln.adain_block(layer_in, sty, 256)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     #Attention Layer at 28x28x256 - No parameters are specified, nor the type of attention layer, so default for now
#     layer_in = Attention()(layer_in)
    
#     layer_in = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
#     #AdaIN

#     layer_in =adaln.adain_block(layer_in, sty, 256)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     ##4 resnet, adain, leakyrelu 0.2
#     #add ResNet Conv2D Layer, 3x3, stride 1, pad same, 128
#     layer_in = residual_module(layer_in, 128)
#     #AdaIN
#     layer_in =adaln.adain_block(layer_in, sty, 128)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     layer_in = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
#     #AdaIN
#     layer_in =adaln.adain_block(layer_in, sty, 128)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     ##5 resnet, adain, leakyrelu 0.2
#     #add ResNet Conv2D Layer, 3x3, stride 1, pad same, 64
#     layer_in = residual_module(layer_in, 64)
#     #AdaIN
#     layer_in =adaln.adain_block(layer_in, sty, 64)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     # ConvTranspose2D Layer, 2x2, stride 2, pad upscale, 64, AdaIN, and leakyReLU 0.2
#     layer_in = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
#     layer_in =adaln.adain_block(layer_in, sty, 64)
#     layer_in = LeakyReLU(0.2)(layer_in)
    
#     layer_out = Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same', activation='tanh')(layer_in)
    
#     generator = Model(inputs = inputs, outputs = layer_out)
    
#     return generator


