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
from keras.layers import Input, Attention, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose

import adaln
from resnet import residual_module


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, IMAGE_SIZE = 96):
    # dataset parameters
     train_path = os.path.join(base_dir, 'train')
     valid_path = os.path.join(base_dir, 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')
     
     return train_gen, val_gen, datagen
 

def combined_model(image_size, latent_dim):
     """
     The combined model of the histopathology generator and discriminator
     
     Args:
         latent_dim: The latent dimension.
         
     Returns:
         The combined model, the generator and the discriminator.
     """
     g = get_generator_histopathology(image_size, latent_dim)
     d = get_discriminator_histopathology(28)
     noise = Input(shape=(latent_dim,))
     g_out = g(noise)
     d.trainable = False
     d_out = d(g_out)
     model = keras.models.Model(inputs=noise, outputs=d_out)
     model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
     return model, g, d
 

def get_discriminator_histopathology(input_size):
   """
   The histopathology discriminator, based on the architecture of the Pathology GAN
   
   Args:
       input_size: The size of the images, as an int (eg. 28 means a 28x28 RGB image is used).
       
   Returns:
       The discriminator model.
   """
   #discriminator = keras.models.Sequential()
   
   #since resnet works with an input layer, make the input to work with resnet
   image_input = Input(shape=(input_size, input_size,3)) #CHECK WHERE THE THREE GOES
   
   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 3
   layer_in = residual_module(image_input, 3)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2
   layer_in = Conv2D(3, kernel_size=(2, 2), strides=(2, 2), padding='downscale')(layer_in)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 32
   layer_in = residual_module(image_input, 32)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2, 32
   layer_in = Conv2D(32, kernel_size=(2, 2), strides=(2, 2), padding='downscale')(layer_in)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 64
   layer_in = residual_module(image_input, 64)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2, 64
   layer_in = Conv2D(64, kernel_size=(2, 2), strides=(2, 2), padding='downscale')(layer_in)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 128
   layer_in = residual_module(image_input, 128)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   #Attention Layer at 28x28x128 - No parameters are specified, nor the type of attention layer, so default for now
   layer_in = Attention()(layer_in)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2, 128
   layer_in = Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding='downscale')(layer_in)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   #resnet conv2d, 3x3, stride 1, pad same, leakyReLu 0.2, 256
   layer_in = residual_module(image_input, 256)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   #conv2d 2x2, stride 2, pad downscale, leakyReLu 0.2, 256
   layer_in = Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding='downscale')(layer_in)
   layer_in = LeakyReLU(0.2)(layer_in)
   
   layer_in = Flatten()(layer_in) #7x7x512
   layer_in = Dense(1024)(layer_in)
   layer_in = LeakyReLU()(layer_in)
   layer_out = Dense(1, activation='leaky_relu')(layer_in)
   
   discriminator = Model(inputs = image_input, outputs = layer_out)
   
   return discriminator


def mapping(noise):
    layer_in = residual_module(noise, 200)
    layer_in = Dense(200)(layer_in)
    layer_in = LeakyReLU(0.2)(layer_in)
    layer_in = residual_module(layer_in, 200)
    layer_in = Dense(200)(layer_in)
    layer_in = LeakyReLU(0.2)(layer_in)
    layer_in = residual_module(layer_in, 200)
    layer_in = Dense(200)(layer_in)
    layer_in = LeakyReLU(0.2)(layer_in)
    layer_in = residual_module(layer_in, 200)
    layer_in = Dense(200)(layer_in)
    layer_in = LeakyReLU(0.2)(layer_in)
    w = Dense(200)(layer_in)
    return w

def get_generator_histopathology(image_size, latent_dim = 300): 
    """
    The histopathology generator, based on the architecture of the Pathology GAN
    
    Args:
        latent_dim: The latent dimension
        
    Returns:
        The generaot model.
    """
    #generator = keras.models.Sequential()
    inputs = Input(shape = (latent_dim,))
    layer_in = Dense(1024, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    layer_in = Reshape([1024,1,1])(layer_in)
    #AdaIN
    #generate the style, beta and gamma
    fil = 1024
    sty = Dense(fil, kernel_initializer = 'he_normal')(Input(shape = [latent_dim]))
    sty = LeakyReLU(0.1)(sty)
    sty = Dense(fil, kernel_initializer = 'he_normal')(sty)
    sty = LeakyReLU(0.1)(sty)

    inp_n = Input(shape = (image_size,image_size, 1))
    noi = [Activation('linear')(inp_n)]
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil, u=False)

    layer_in = LeakyReLU()(layer_in) #Ran with default becasue 0.2 is not specified, default is 0.3
    
    layer_in = Dense(12544, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(layer_in)
    #layer_in = Reshape([12544,1,1])(layer_in)
    #AdaIN
    fil = 12544
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU()(layer_in) #Ran with default becasue 0.2 is not specified, default is 0.3
    
    layer_in = Reshape((256, 7, 7))(layer_in)
    
    ##1 resnet, adain, leakyrelu 0.2
    #ResNet Conv2D Layer, 3x3, stride 1, pad same, 256
    layer_in = residual_module(layer_in, 256)
    #AdaIN
    ### necessary step for the adain
    fil = 256
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    layer_in = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
    #AdaIN
    #generate the style, beta and gamma
    fil = 256
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    ##2 resnet, adain, leakyrelu 0.2
    #add ResNet Conv2D Layer, 3x3, stride 1, pad same, 512
    layer_in = residual_module(layer_in, 512)
    #AdaIN
    #generate the style, beta and gamma
    fil = 512
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    layer_in = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
    #AdaIN
    #generate the style, beta and gamma
    fil = 256
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    ##3 resnet, attention layer, adain, leakyrelu 0.2
    #add ResNet Conv2D Layer, 3x3, stride 1, pad same, 256
    layer_in = residual_module(layer_in, 256)
    #AdaIN
    #generate the style, beta and gamma
    fil = 256
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    #Attention Layer at 28x28x256 - No parameters are specified, nor the type of attention layer, so default for now
    layer_in = Attention()(layer_in)
    
    layer_in = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
    #AdaIN
    #generate the style, beta and gamma
    fil = 256
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    ##4 resnet, adain, leakyrelu 0.2
    #add ResNet Conv2D Layer, 3x3, stride 1, pad same, 128
    layer_in = residual_module(layer_in, 128)
    #AdaIN
    #generate the style, beta and gamma
    fil = 128
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    layer_in = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
    #AdaIN
    #generate the style, beta and gamma
    fil = 128
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    ##5 resnet, adain, leakyrelu 0.2
    #add ResNet Conv2D Layer, 3x3, stride 1, pad same, 64
    layer_in = residual_module(layer_in, 64)
    #AdaIN
    #generate the style, and input noise
    fil = 64
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    # ConvTranspose2D Layer, 2x2, stride 2, pad upscale, 64, AdaIN, and leakyReLU 0.2
    layer_in = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2,2), padding='upscale')(layer_in) #CHECK IF UPSCALE WORKS
    #AdaIN
    #generate the style, beta and gamma
    fil = 64
    layer_in =adaln.adain_block(layer_in, sty, noi[0], fil)
    layer_in = LeakyReLU(0.2)(layer_in)
    
    layer_out = Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same', activation='tanh')(layer_in)
    
    generator = Model(inputs = inputs, outputs = layer_out)
    
    return generator


def generate_latent_points(latent_dim, n_samples):
    """
    Generate points in latent space as input for the generator
    
    Args:
        latent_dim: The latent dimension.
        n_samples: The number of samples generated
        
    Returns:
        An array of the generated points.
    """
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input