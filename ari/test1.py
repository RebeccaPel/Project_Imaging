# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:25:50 2022

@author: Ari
"""

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import Orthogonal

import keras
keras.backend.set_image_data_format('channels_first')
from keras import backend as K

from keras.models import Model
from keras.layers import Layer, Dense, Dropout, Flatten, Conv2D, Input, Reshape, Attention, Activation, Conv2DTranspose, UpSampling2D, add, LayerNormalization, MultiHeadAttention
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.callbacks import TensorBoard
from keras.datasets.mnist import load_data

import pickle
import gzip
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import pylab
from IPython import display

import tensorflow as tf
import keras
keras.backend.set_image_data_format('channels_last')

import time
import random
from google.colab import files
from google.colab import widgets
random.seed(0)

from utilsGAN import get_pcam_generators, saveModels, plotImagesPatchCamelyon, plotGeneratedImagesPatchCamelyon, crop
from resnet import ResBlock
from functionsGAN import get_discriminator_histopathology, get_generator_histopathology

# Relevant variables
IMAGE_SIZE = 96
batch_size = 500

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input = Input(input_shape)
train_gen, val_gen = get_pcam_generators(r'D:\Ari\Uni\TUE\8P361', batch_size)

#crop_train_gen = crop(train_gen, 0.333)
#crop_val_gen = crop(val_gen, 0.333)


latent_dim = 200

discriminator = get_discriminator_histopathology()
generator = get_generator_histopathology()

discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.8))
generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.8))

discriminator.trainable = False
z = keras.layers.Input(shape=(latent_dim,))
x = generator(z)
D_G_z = discriminator(x)
gan = keras.models.Model(inputs=z, outputs=D_G_z)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.8))


d_losses = []
g_losses = []

epochs = 100

# map pixel values to the [-1, 1] range to be compatible with tanh activation function

batch_count = 144000 // (batch_size*10)
for e in range(epochs):
    for b in range(batch_count):
        start_time = time.time()
        # Get a random set of input noise and images
        noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
        # image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
        image_batch = tf.image.central_crop(next(train_gen)[0], 1)
        print(image_batch.shape)

        # Generate some fake histopathology images using the generator
        generated_images = generator.predict(noise)
        print(generated_images.shape)
        # Concatenate the fake and real images
        X = np.concatenate([image_batch, generated_images])

        # Labels for generated and real data
        y_dis = np.zeros(2 * batch_size)
        # Set reference to 1 for real samples
        y_dis[:batch_size] = 1

        # Train discriminator with this batch of samples
        discriminator.trainable = True
        try:
            d_loss = discriminator.train_on_batch(X, y_dis)
        except ValueError:
            print(image_batch.shape, noise.shape)
            continue
        # Train generator with a new batch of generated samples
        noise = np.random.normal(0, 1, size=[batch_size, latent_dim])

        # From the generator's perspective, the discriminator should predict
        # ones for all samples
        y_gen = np.ones(batch_size)

        # Freeze the discriminator part
        discriminator.trainable = False

        # Train the GAN to predict ones
        g_loss = gan.train_on_batch(noise, y_gen)
        end_time = time.time()
        time_1_batch = end_time - start_time
        # Store loss of most recent batch from this epoch
        if int(b * 100 / batch_count) % 5 == 0:
            print(str(int(b * 100 / batch_count)) + '% done with epoch number' + str(e))
            print('eta: ' + str(
                (epochs - (e + 1)) * time_1_batch * batch_count + (time_1_batch * batch_count - b)) + ' seconds')
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if e % 5 == 0:
        noise = np.random.normal(0, 1, size=[100, latent_dim])
        generatedImages = generator.predict(noise)
        generatedImages = generatedImages.reshape(100, 96, 96, 3)
        plotImagesPatchCamelyon(generatedImages,
                                title='Epoch {}'.format(e))  # map pixel values to the [0, 1] range for plotting
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(0.001)
        saveModels(e)

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(d_losses)
plt.title('Discriminator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(g_losses)
plt.title('Generator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
print(d_losses, g_losses)
plotGeneratedImagesPatchCamelyon(epochs, generator, discriminator)

## JUSTIN's stuff

# def add_noise_2(model):
#     for layer in model.trainable_weights:
#         noise = np.random.normal(loc=0.0, scale=0.02, size=layer.shape)
#         layer.assign_add(noise)


# def transfer_classifier(transfer_source):
#     transfer_source.trainable = False
#     inputs = transfer_source.inputs
#     # take the output from the second to last layer of the full model and
#     # replace the last layer with a sigmoid 1-unit dense layer for classification of the PCAM set
#     outputs = Dense(1, activation='sigmoid')(transfer_source.layers[-2].output)
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999),
#                   metrics=['accuracy'])
#     #add_noise_2(model)
#     return model


# def classifier():
#     model = get_discriminator_histopathology()
#     model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999),
#                   metrics=['accuracy'])
#     return model


# steps = 144000 // batch_size
# transfer = transfer_classifier(discriminator)
# standard = classifier()
# model_name_1 = 'transfer'
# model_name_2 = "standard"
# tensorboard_1 = TensorBoard("logs/" + model_name_1)
# tensorboard_2 = TensorBoard("logs/" + model_name_2)
# # do initial training of the transfer model
# transfer.fit(crop_train_gen, batch_size=128, steps_per_epoch=steps, epochs=3, verbose=1, callbacks=[tensorboard_1])
# # unfreeze the layers of the transfer model and finetune
# for layer in transfer.layers:
#     layer.trainable = True
# transfer.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999),
#               metrics=['accuracy'])
# transfer.fit(crop_train_gen, batch_size=128, steps_per_epoch=steps, epochs=3, verbose=1, callbacks=[tensorboard_1])
# standard.fit(crop_train_gen, batch_size=128, steps_per_epoch=steps, epochs=3, verbose=1, callbacks=[tensorboard_2])
# standard.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999),
#               metrics=['accuracy'])
# standard.fit(crop_train_gen, batch_size=128, steps_per_epoch=steps, epochs=3, verbose=1, callbacks=[tensorboard_2])
# transfer.summary()
# standard.summary()
# transfer.evaluate(crop_val_gen, steps=steps, batch_size=128)
# standard.evaluate(crop_val_gen, steps=steps, batch_size=128)