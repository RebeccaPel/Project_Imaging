# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:22:52 2022

@author: Ari
"""

import gzip
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from np.random import randn

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 96
batch_size = 500

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, IMAGE_SIZE = 96):
    # dataset parameters
     train_path = os.path.join(base_dir,'train+val', 'train')
     valid_path = os.path.join(base_dir,'train+val', 'valid')

    #Check why it's here and if it's needed
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

def loadPatchCamelyon(path):
    f = gzip.open(path, 'rb')
    train_set = pickle.load(f, encoding='latin1')
    f.close()
    return train_set


def crop(gen, percentage):
    for x in gen:
        yield tf.image.central_crop(x[0], percentage), x[1]


def saveModels(generator, discriminator, epoch):
    generator.save('gan_generator_epoch_{}.h5'.format(epoch))
    discriminator.save('gan_discriminator_epoch_{}.h5'.format(epoch))

def plotImagesPatchCamelyon(images, dim=(10, 10), figsize=(10, 10), title=''):
    images = images.astype(np.float32) * 0.5 + 0.5
    plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
    
def plotGeneratedImagesPatchCamelyon(epoch, generator, discriminator, latent_dim = 200, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 32, 32, 3)
    generatedImages = generatedImages.astype(np.float32) * 0.5 + 0.5

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle('Epoch {}'.format(epoch))
    plt.show()

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