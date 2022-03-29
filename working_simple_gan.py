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
from keras import backend as K
from keras.layers import Layer
from keras.layers import InputSpec
from keras import initializers, regularizers, constraints

IMAGE_SIZE = 32
batch_size = 128

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input = Input(input_shape)



# From a PR that is not pulled into Keras
# https://github.com/fchollet/keras/pull/3677
# I updated the code to work on Keras 2.x

class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.
    # Example
    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)
        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```
    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.
    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
    keras.models.save_model(generator, 'gan_generator_epoch_proper_disc{}.h5'.format(epoch))
    keras.models.save_model(discriminator, 'gan_discriminator_epoch_proper_disc{}.h5'.format(epoch))
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
        in_shape=(32, 32, 3,)):
    inputs = Input(shape=in_shape)
    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=in_shape,
               kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    # input shape accounts for the three channels
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = MinibatchDiscrimination(16, 3)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=inputs, outputs=outputs)
    return discriminator


def get_generator_histopathology():
    inputs = Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 128, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 128))(x)
    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    outputs = Conv2D(3, kernel_size=(3, 3), padding='same', activation='tanh')(x)  # 3 filters for 3 channels (rgb)
    generator = Model(inputs=inputs, outputs=outputs)
    return generator


latent_dim = 400

discriminator = get_discriminator_histopathology()
generator = get_generator_histopathology()

discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.5, beta_2=0.999))
generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999))

discriminator.trainable = False
z = keras.layers.Input(shape=(latent_dim,))
x = generator(z)
D_G_z = discriminator(x)
gan = keras.models.Model(inputs=z, outputs=D_G_z)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999))


d_losses_1 = []
d_losses_2 = []
g_losses = []
batch_times = []
epochs = 200
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject')
val_gen_crop = crop(val_gen, 0.33333)
# map pixel values to the [-1, 1] range to be compatible with tanh activation function
generator.summary()
discriminator.summary()
batch_count = 144000 // batch_size
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
        X_real = image_batch
        X_fake = generated_images

        # Labels for generated and real data
        y_real = np.zeros(batch_size)
        # smooth labels
        y_real[:] = 0.9
        print(y_real)
        y_fake = np.zeros(batch_size)

        # Train discriminator with this batch of samples
        # training has to be seperated between real and fake images for the minibatch discrimination to work
        discriminator.trainable = True
        try:
            d_loss_1 = discriminator.train_on_batch(X_real, y_real)
            d_loss_2 = discriminator.train_on_batch(X_fake, y_fake)
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

        # pretrain the discriminator exclusively for the first few batches
        if e == 0 and b < 20:
            continue
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
    d_losses_1.append(d_loss_1)
    d_losses_2.append(d_loss_2)
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
plt.plot(d_losses_2)
plt.title('Discriminator loss on fake images')
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
