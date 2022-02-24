import tensorflow as tf
import keras
from keras.datasets.mnist import load_data
from keras import models
from keras import layers
from keras import activations
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import Flatten
from numpy.random import randn

(trainX, trainy), (testX, testy) = load_data()
trainX = (trainX / 255.0).astype(np.float32)
testX = (testX / 255.0).astype(np.float32)
trainy = tf.keras.utils.to_categorical(trainy, 10)
testy = tf.keras.utils.to_categorical(testy, 10)
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)


def discriminator(input_size):
    input_shape = (input_size, input_size, 1)
    model = keras.models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(0.02))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(0.02))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation=activations.sigmoid))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return model


def generator(latent_dim):
    model = keras.Sequential()
    model.add(layers.Dense(7 * 7 * 128, input_shape=(latent_dim,)))
    model.add(layers.LeakyReLU(0.02))
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(128, (4, 4), padding='same', strides=(2, 2)))
    model.add(layers.LeakyReLU(0.02))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation=activations.leaky_relu))
    model.add(layers.LeakyReLU(0.02))
    model.add(layers.Conv2D(1, (7, 7), padding='same', activation=activations.tanh))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input


def combined_model(latent_dim):
    g = generator(latent_dim)
    d = discriminator(28)
    noise = layers.Input(shape=(latent_dim,))
    g_out = g(noise)
    d.trainable = False
    d_out = d(g_out)
    model = keras.models.Model(inputs=noise, outputs=d_out)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return model, g, d

epochs = 10
latent_dim = 100
n_samples = 125
generator_losses = []
discriminator_losses = []
batch_count = int(trainX.shape[0] / n_samples)
gan, generator, discriminator = combined_model(latent_dim)

for e in range(epochs):
    for b in range(batch_count):
        noise = generate_latent_points(latent_dim, n_samples)
        image_batch = trainX[np.random.randint(0, trainX.shape[0], size=n_samples)]
        generated = generator(noise)
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

gan.save(r'C:\Users\justi\PycharmProjects\pythonProject\model')

def transfer_classifier(transfer_source):
    transfer_source.trainable = True
    model = transfer_source
    model.pop()
    # take the output from the second to last layer of the full model
    model.add(layers.Dense(10,activation=activations.softmax))
    # replace the last layer with a softmax 10-unit dense layer for classification of the mnist set
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  metrics=['accuracy'])
    return model


def classifier():
    input_shape = (28, 28, 1)
    model = keras.models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(0.02))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(0.02))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(10,activation=activations.softmax))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model



transfer = transfer_classifier(discriminator)
standard = classifier()
model_name_1 = 'transfer'
model_name_2 = "standard"
tensorboard_1 = TensorBoard("logs/" + model_name_1)
tensorboard_2 = TensorBoard("logs/" + model_name_2)
transfer.fit(trainX, trainy, batch_size=128, epochs=3, verbose=1, callbacks=[tensorboard_1])
standard.fit(trainX, trainy, batch_size=128, epochs=3, verbose=1, callbacks=[tensorboard_2])
transfer.summary()
standard.summary()
transfer.evaluate(testX, testy, batch_size=128)
standard.evaluate(testX, testy, batch_size=128)
