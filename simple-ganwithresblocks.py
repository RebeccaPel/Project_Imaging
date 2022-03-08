import tensorflow as tf
import keras
import os
import time
import numpy as np
from keras.datasets.mnist import load_data
from keras import models
from keras import layers
from keras import activations
from tensorflow_addons.layers import SpectralNormalization
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
from keras.layers import Input
from numpy.random import randn
from keras.layers import Activation, Reshape, Lambda, dot, add
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.initializers import Orthogonal

BATCH_SIZE = 256


def get_pcam_generators(base_dir, train_batch_size=BATCH_SIZE, val_batch_size=BATCH_SIZE):
    # dataset parameters
    train_path = os.path.join(base_dir, 'train+val', 'train')
    valid_path = os.path.join(base_dir, 'train+val', 'valid')

    # instantiate data generators
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=train_batch_size,
                                            class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=val_batch_size,
                                          class_mode='binary')

    return train_gen, val_gen


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input = Input(input_shape)
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject')
(trainX, trainy), (testX, testy) = load_data()
trainX = (trainX / 255.0).astype(np.float32)
testX = (testX / 255.0).astype(np.float32)
trainy = tf.keras.utils.to_categorical(trainy, 10)
testy = tf.keras.utils.to_categorical(testy, 10)
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)


# code based on tensorflow tutorial on making custom layers
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


def generator(latent_dim):
    inputs = keras.Input(shape=(latent_dim,))
    # start with a 4*4*32*16 vector
    dense = layers.Dense(4 * 4 * 32 * 16, activation=activations.leaky_relu, kernel_initializer=Orthogonal)
    # output of the dense layer
    x = dense(inputs)
    x = layers.Reshape((4, 4, 512))(x)
    # shape = 4,4,512
    x = ResBlockUp([384, 256])(x)
    # shape = 8, 8, 256
    x = ResBlockUp([192, 128])(x)
    # shape = 16, 16, 128
    # x = non_local_block(x, compression=2, mode='embedded')
    x = ResBlockUp([96, 64])(x)
    # shape = 32, 32, 64
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation=activations.tanh)(x)
    # shape = 32,32,3 (32x32 rgb image)
    model = keras.Model(inputs=inputs, outputs=outputs, name="generator")
    return model


def discriminator(input_size, training=True):
    inputs = keras.Input(shape=(input_size, input_size, 3))
    # shape = (32,32,3)
    x = ResBlockDown([32, 32], (3, 3))(inputs)
    # shape = (16,16,32)
    # x = non_local_block(x, compression=2, mode='embedded')
    x = ResBlockDown([64, 64], (3, 3))(x, training)
    # shape = (8,8,64)
    x = ResBlockDown([128, 128], (3, 3))(x, training)
    # shape = (4,4,128)
    x = ResBlockDown([256, 256], (3, 3))(x, training)
    # shape = (2,2,256)
    x = ResBlock([256, 256], (3, 3))(x, training)
    x = tf.nn.relu(x)
    # GlobalSumPooling for (2x2x256) images
    x = layers.GlobalAveragePooling2D()(x) * 4
    outputs = Dense(1, activation=activations.tanh)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="discriminator")
    return model


# def discriminator(input_size):
#     input_shape = (input_size, input_size, 3)
#     model = keras.models.Sequential(name='transfer')
#     model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
#     model.add(layers.LeakyReLU(0.02))
#     model.add(layers.Dropout(0.3))
#     model.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
#     model.add(layers.LeakyReLU(0.02))
#     model.add(layers.Dropout(0.3))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(1, activation=activations.sigmoid))
#     model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
#     return model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    z_input = np.random.normal(size=latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input


def combined_model(latent_dim):
    g = generator(latent_dim)
    d = discriminator(32)
    d.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0, beta_2 = 0.999))
    g.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0,
                                                                             beta_2 = 0.999))
    noise = layers.Input(shape=(latent_dim,))
    g_out = g(noise)
    d.trainable = False
    d_out = d(g_out)
    model = keras.models.Model(inputs=noise, outputs=d_out)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0,
                                                                                 beta_2 = 0.999))
    return model, g, d


epochs = 5
latent_dim = 200
n_samples = 256
generator_losses = []
discriminator_losses = []
# if you want to use the full training set
# batch_count = len(train_gen)
# for testing I use a smaller dataset
batch_count = len(train_gen) // 15
gan, generator, discriminator = combined_model(latent_dim)

for e in range(epochs):
    for b in range(batch_count):
        start_time = time.time()
        noise = generate_latent_points(latent_dim, n_samples)
        # image_batch = trainX[np.random.randint(0, trainX.shape[0], size=n_samples)]
        # crop into the center 32x32 pixels
        image_batch = tf.image.central_crop(train_gen.next()[0], 0.3333)
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
        end_time = time.time()
        time_1_batch = end_time - start_time
        if int(b * 100 / batch_count) % 5 == 0:
            print(str(int(b * 100 / batch_count)) + '% done with epoch number' + str(e))
            print('eta: ' + str(
                (epochs - (e + 1)) * time_1_batch * batch_count + (time_1_batch * batch_count - b)) + ' seconds')
    noise = generate_latent_points(latent_dim, 16)
    predictions = generator(noise)
    predictions = np.reshape(predictions, (16, 32, 32, 3))
    predictions = predictions / 2 + 0.5
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')
    plt.show()
    generator_losses.append(generator_loss)
    discriminator_losses.append(discriminator_loss)
    print(discriminator_losses, generator_losses)

gan.save(r'C:\Users\justi\PycharmProjects\pythonProject\model')


def transfer_classifier(transfer_source):
    transfer_source.trainable = True
    model = transfer_source
    model.pop()
    # take the output from the second to last layer of the full model
    model.add(layers.Dense(10, activation=activations.softmax))
    # replace the last layer with a softmax 10-unit dense layer for classification of the mnist set
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  metrics=['accuracy'])
    return model


def classifier():
    input_shape = (28, 28, 1)
    model = keras.models.Sequential(name="standard")
    model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(0.02))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(0.02))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation=activations.softmax))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  metrics=['accuracy'])
    return model


transfer = transfer_classifier(discriminator)
standard = classifier()
model_name_1 = 'transfer'
model_name_2 = "standard"
tensorboard_1 = TensorBoard("logs/" + model_name_1)
tensorboard_2 = TensorBoard("logs/" + model_name_2)
# transfer.fit(train_gen, batch_size=128, epochs=3, verbose=1, callbacks=[tensorboard_1])
# standard.fit(trainX, trainy, batch_size=128, epochs=3, verbose=1, callbacks=[tensorboard_2])
# transfer.summary()
# standard.summary()
# transfer.evaluate(testX, testy, batch_size=128)
# standard.evaluate(testX, testy, batch_size=128)
