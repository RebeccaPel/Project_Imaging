import gzip
import os
import pickle
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from keras import layers
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 32
batch_size = 250

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input = Input(input_shape)


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


class ResBlockDown(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResBlockDown, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        self.conv2a = Conv2D(filters[0], (1, 1), padding='same',
                             kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2a = keras.layers.BatchNormalization()

        self.conv2b = Conv2D(filters[1], kernel_size, padding='same', strides=(2, 2),
                             kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2b = keras.layers.BatchNormalization()
        # self.av2b = keras.layers.AvgPool2D((2, 2))

        self.conv2in = Conv2D(filters[1], 1, padding='same', strides=(2, 2),
                              kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        # self.av2in = keras.layers.AvgPool2D((2, 2))

    def get_config(self):
        config = super(ResBlockDown, self).get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, input_tensor):
        x = self.conv2a(input_tensor)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = tf.nn.relu(x)
        # x = self.av2b(x)

        skip = self.conv2in(input_tensor)
        # skip = self.av2in(skip)
        x += skip
        return x


class ResBlockUp(keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), **kwargs):
        super(ResBlockUp, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.up2a = keras.layers.UpSampling2D((2, 2))

        self.conv2a = Conv2DTranspose(filters[0], kernel_size, padding='same', strides=(2, 2),
                                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2a = keras.layers.BatchNormalization()

        self.conv2b = Conv2D(filters[1], kernel_size, padding='same',
                             kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.bn2b = keras.layers.BatchNormalization()

        self.conv2in = Conv2DTranspose(filters[1], 1, padding='same', strides=(2, 2),
                                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.up2in = keras.layers.UpSampling2D((2, 2))

    def get_config(self):
        config = super(ResBlockUp, self).get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, input_tensor, training=True):
        x = self.bn2a(input_tensor, training=training)
        x = tf.nn.relu(x)
        # x = self.up2a(x)
        x = self.conv2a(x)

        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        # skip = self.up2in(input_tensor)
        skip = self.conv2in(input_tensor)
        x += skip
        return x


def get_pcam_generators(base_dir, train_batch_size=batch_size, val_batch_size=batch_size):
    # dataset parameters
    train_path = os.path.join(base_dir, 'train+val', 'train')
    valid_path = os.path.join(base_dir, 'train+val', 'valid')

    # instantiate data generators
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(32, 32),
                                            batch_size=train_batch_size,
                                            class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=(32, 32),
                                          batch_size=train_batch_size,
                                          class_mode='binary')

    return train_gen, val_gen


def loadPatchCamelyon(path):
    f = gzip.open(path, 'rb')
    train_set = pickle.load(f, encoding='latin1')
    f.close()
    return train_set


def crop(gen, percentage):
    for x in gen:
        yield tf.image.central_crop(x[0], percentage), x[1]


def saveModels(epoch):
    generator.trainable = True
    discriminator.trainable = True
    keras.models.save_model(generator, 'gan_generator_epoch_{}.h5'.format(epoch))
    keras.models.save_model(discriminator, 'gan_discriminator_epoch_{}.h5'.format(epoch))
    generator.trainable = False
    discriminator.trainable = False


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


def plotGeneratedImagesPatchCamelyon(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
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


def get_discriminator_histopathology(
        in_shape=(32, 32, 3)):
    inputs = Input(shape=in_shape)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=in_shape,
               kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    # input shape accounts for the three channels
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=inputs, outputs=outputs)
    return discriminator


def get_discriminator_histopathology_res(in_shape=(32, 32, 3)):
    inputs = Input(shape=in_shape)
    x = ResBlockDown([16, 16], kernel_size=(4, 4))(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = ResBlockDown([32, 32], kernel_size=(4, 4))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = ResBlockDown([32, 32], kernel_size=(4, 4))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = ResBlockDown([64, 64], kernel_size=(4, 4))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = ResBlockDown([64, 64], kernel_size=(4, 4))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = ResBlock([64, 64], kernel_size=(4, 4))(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=inputs, outputs=outputs)
    return discriminator


def get_generator_histopathology():
    inputs = Input(shape=(latent_dim,))
    x = Dense(64 * 4 * 4, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Reshape((4, 4, 64))(x)
    x = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    # x = Conv2D(64, kernel_size=(4, 4), padding='same')(x)
    # x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    outputs = Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='tanh')(x)
    generator = Model(inputs=inputs, outputs=outputs)
    return generator


def get_generator_histopathology_res():
    inputs = Input(shape=(latent_dim,))
    x = Dense(128 * 4 * 4, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Reshape((4, 4, 128))(x)
    x = ResBlockUp([128, 128], (4, 4))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = ResBlockUp([64, 64], (4, 4))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = ResBlockUp([32, 32], (4, 4))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2D(3, kernel_size=(4, 4), padding="same")(x)
    outputs = layers.Activation(activation='tanh')(x)
    generator = Model(inputs=inputs, outputs=outputs)
    return generator


latent_dim = 400

discriminator = get_discriminator_histopathology_res()
generator = get_generator_histopathology()

discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9))
generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

discriminator.trainable = False
z = keras.layers.Input(shape=(latent_dim,))
x = generator(z)
D_G_z = discriminator(x)
gan = keras.models.Model(inputs=z, outputs=D_G_z)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

d_losses = []
g_losses = []
batch_times = []
epochs = 500
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject')
val_gen_crop = crop(val_gen, 0.33333)
# map pixel values to the [-1, 1] range to be compatible with tanh activation function
generator.summary()
discriminator.summary()
batch_count = 144000 // batch_size // 10
for e in range(epochs):
    for b in range(batch_count):
        start_time = time.time()
        # Get a random set of input noise and images
        noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
        # image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
        image_batch = next(train_gen)[0]

        # Generate some fake histopathology images using the generator
        generated_images = generator.predict(noise)

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
        batch_times.append(end_time - start_time)
        # Store loss of most recent batch from this epoch
        if int(b * 100 / batch_count) % 2 == 0:
            if b > 2:
                time_1_batch = np.mean(batch_times[2:])
            else:
                time_1_batch = batch_times[0]
            n = int(b * 100 / batch_count)
            n_batches_left = ((epochs - (e + 1)) * batch_count) + (batch_count - (b + 1))
            eta = time_1_batch * n_batches_left
            print('\r%d%% done with epoch number %d, eta: %d seconds, batch time: %f' % (n, e + 1, eta, time_1_batch),
                  end="")
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if e % 5 == 0:
        noise = np.random.normal(0, 1, size=[100, latent_dim])
        generatedImages = generator.predict(noise)
        generatedImages = generatedImages.reshape(100, 32, 32, 3)
        plotImagesPatchCamelyon(generatedImages,
                                title='Epoch {}'.format(e))  # map pixel values to the [0, 1] range for plotting
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(0.01)
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
plotGeneratedImagesPatchCamelyon(epochs)
