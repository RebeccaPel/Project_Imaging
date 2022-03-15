# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 08:27:28 2022

@author: Ari
"""
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from functionsGAN import get_pcam_generators, generate_latent_points, combined_model, mapping

from keras.layers import Dense, Dropout, Flatten, Conv2D, Input
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.callbacks import TensorBoard
from keras.datasets.mnist import load_data

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input = Input(input_shape)
train_gen, val_gen, datagen = get_pcam_generators(r'D:\Ari\Uni\TUE\8P361')

def crop(gen, percentage):
    for x in gen:
        yield tf.image.central_crop(x[0], percentage), x[1]

crop_train_gen = crop(train_gen, 0.333)
crop_val_gen = crop(val_gen, 0.333)

(trainX, trainy), (testX, testy) = load_data()
trainX = (trainX / 255.0).astype(np.float32)
testX = (testX / 255.0).astype(np.float32)
trainy = tf.keras.utils.to_categorical(trainy, 10)
testy = tf.keras.utils.to_categorical(testy, 10)
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

epochs = 1
latent_dim = 200
n_samples = 5
generator_losses = []
discriminator_losses = []
batch_count = int(trainX.shape[0] / n_samples)
gan, generator, discriminator = combined_model(32, latent_dim)

for e in range(epochs):
    for b in range(batch_count):
        noise = generate_latent_points(latent_dim, n_samples)
        w_noise = mapping(noise)
        image_batch = trainX[np.random.randint(0, trainX.shape[0], size=n_samples)]
        generated = generator(w_noise)
        X = np.concatenate([image_batch, generated])
        y_dis = np.zeros(2 * n_samples)
        y_dis[:n_samples] = 0.9
        discriminator.trainable = True
        discriminator_loss = discriminator.train_on_batch(X, y_dis)
        noise = generate_latent_points(latent_dim, n_samples)
        y_gen = np.ones(n_samples)
        discriminator.trainable = False
        generator_loss = gan.train_on_batch(noise, y_gen)
        if int(b * 100 / batch_count) % 20 == 0:
            print(str(int(b * 100 / batch_count)) + '% done with epoch number' + str(e))
    generator_losses.append(generator_loss)
    discriminator_losses.append(discriminator_loss)
    noise = generate_latent_points(latent_dim, 16)
    predictions = generator(noise)
    predictions = np.reshape(predictions, (16, 28, 28))
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i], cmap='gray')
        plt.axis('off')
    plt.show()

#gan.save(r'C:\Users\justi\PycharmProjects\pythonProject\model')
gan.save(r'\model')

def transfer_classifier(transfer_source):
    transfer_source.trainable = True
    model = transfer_source
    model.pop()
    # take the output from the second to last layer of the full model
    model.add(Dense(10))
    model.add(Softmax())
    # replace the last layer with a softmax 10-unit dense layer for classification of the mnist set
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  metrics=['accuracy'])
    return model


def classifier():
    input_shape = (28, 28, 1)
    model = keras.models.Sequential()
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(0.02))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(0.02))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model



transfer = transfer_classifier(discriminator)
standard = classifier()
model_name_1 = 'transfer'
model_name_2 = "standard"
tensorboard_1 = TensorBoard("logs/" + model_name_1)
tensorboard_2 = TensorBoard("logs/" + model_name_2)
transfer.fit(trainX, trainy, batch_size=16, epochs=3, verbose=1, callbacks=[tensorboard_1])
standard.fit(trainX, trainy, batch_size=16, epochs=3, verbose=1, callbacks=[tensorboard_2])
transfer.summary()
standard.summary()
transfer.evaluate(testX, testy, batch_size=16)
standard.evaluate(testX, testy, batch_size=16)
