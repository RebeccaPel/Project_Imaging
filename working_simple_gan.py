import os
import time
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from keras import layers
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers import Layer
from keras.layers import InputSpec
from keras import initializers, regularizers, constraints

# code source: https://github.com/forcecore/Keras-GAN-Animeface-Character/blob/master/discrimination.py
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
        return input_shape[0], input_shape[1] + self.nb_kernels

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


class UpSampleBlock(Layer):
    """
    Convolves, upsamples and applies leaky relu activation to the input.
    Inherits from the keras.layers.Layer class
    # Arguments:
    dimension: number of filters used in the convolutional layer.
    kernel_size: size of the convolutional kernel
    """

    def __init__(self, dimension, kernel_size=(3, 3), **kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        self.dimension = dimension
        self.kernel_size = kernel_size
        # Conv2Dtranspose let to checkerboarding, this could be fixed by changing kernel size.
        # But this is not desirable.
        self.conv2d = Conv2D(dimension, kernel_size=kernel_size, padding='same')
        self.upsample = UpSampling2D(size=(2, 2))
        self.leakyrelu = LeakyReLU(0.2)
        self.bn = BatchNormalization(momentum=0.3)

    def get_config(self):
        config = super(UpSampleBlock, self).get_config()
        config.update({
            'dimension': self.dimension,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self, input):
        x = self.upsample(input)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x


def get_pcam_generators(base_dir, image_size,  train_batch_size=128, val_batch_size=128):
    """
    Creates keras ImageDataGenerator objects of the datasets. Assigns labels to images based on the folder they're
    contained in.
    Args:
        base_dir: path to the directory containing the dataset. Dataset directories should have the following structure:
         'base_dir\train+val\' followed by either 'train' or 'val' (training and validation set respectively) and either
         '0' or '1' (class labels) folders. Dataset images should be appropriately  placed into their respective class
         label folders
        train_batch_size: batch size to be used for the training set
        val_batch_size: batch size to be used for the validation set

    Returns:
        train_gen: generator for the training set
        val_gen: generator for the validation set

    """
    # dataset parameters
    train_path = os.path.join(base_dir, 'train')
    valid_path = os.path.join(base_dir, 'valid')

    # instantiate data generators
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=image_size,
                                            batch_size=train_batch_size,
                                            class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=image_size,
                                          batch_size=val_batch_size,
                                          class_mode='binary')

    return train_gen, val_gen


def saveModels(discriminator, generator, epoch, name):
    """
    Saves the generator and discriminator models to a keras compatible .h5 file.
    Adapted from: assignments for this course (need better sourcing)

    Args:
        epoch: The training epoch the model is currently on
        name: Name of the model

    Returns: None

    """
    generator.trainable = True
    discriminator.trainable = True
    keras.models.save_model(generator, 'gan_generator_epoch_{name}_{epoch}.h5'.format(name=name, epoch=epoch))
    keras.models.save_model(discriminator, 'gan_discriminator_epoch_{name}_{epoch}.h5'.format(name=name, epoch=epoch))
    generator.trainable = False
    discriminator.trainable = False


def plotImages(images, dim=(10, 10), figsize=(10, 10), title=''):
    """
    Plot a number of images in a grid.
    Source: assignments for this course (need better sourcing)
    Args:
        images: images to be plotted, channels-last.
        dim: tuple containing the dimensions for the image grid
        figsize: tuple containing width, height of the grid in inches
        title: title to be displayed above the image grid.

    Returns: None

    """
    images = images.astype(np.float32) * 0.5 + 0.5
    plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


def plotGeneratedImagesPatchCamelyon(epoch, generator, latent_dim=500, examples=100, dim=(10, 10), figsize=(10, 10)):
    """
    Generates and plots images from a GAN generator network in a grid.
    Args:
        generator: generator network to be used for generating the images
        latent_dim: dimensionality of the input space.
        epoch: training epoch that the images originate from
        examples: number of images to be generated
        dim: tuple containing the dimensions for the image grid
        figsize: tuple containing width, height of the grid in inches


    Returns: None

    """
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
    """
    Constructs and returns the discriminator network used in the GAN.
    Args:
        in_shape: shape of the input tensor, channels-last.

    Returns:
        discriminator: the discriminator network, not yet compiled.
    """
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
    x = MinibatchDiscrimination(5, 3)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=inputs, outputs=outputs)
    return discriminator


def get_generator_histopathology(latent_dim):
    """
    Constructs and returns the generator network to be used in the GAN.
    Returns:
        generator: the generator network, not yet compiled.

    """
    inputs = Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 128, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 128))(x)
    x = UpSampleBlock(128)(x)
    x = UpSampleBlock(64)(x)
    x = UpSampleBlock(64)(x)
    outputs = Conv2D(3, kernel_size=(3, 3), padding='same', activation='tanh')(x)  # 3 filters for 3 channels (rgb)
    generator = Model(inputs=inputs, outputs=outputs)
    return generator


def get_gan(input_size=(32, 32, 3,), latent_dim=500, disc_lr=0.0004, gen_lr=0.0001):
    """

    Args:
        input_size: size of the discriminator input, channels last.
        latent_dim: dimensionality of the input space
        disc_lr: learning rate for the Adam optimizer for the discriminator
        gen_lr: learning rate for the Adam optimizer for the generator

    Returns:
        gan: the compiled combined gan model
        discriminator: the compiled discriminator model
        generator: the compiled generator model

    """
    # Get the component networks
    discriminator = get_discriminator_histopathology(input_size)
    generator = get_generator_histopathology(latent_dim)
    # Compile the component networks
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=0.5, beta_2=0.999))
    generator.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=0.5, beta_2=0.999))
    # Make sure the discriminator does not update when training the generator.
    discriminator.trainable = False
    # create input layer
    z = keras.layers.Input(shape=(latent_dim,))
    x = generator(z)
    D_G_z = discriminator(x)
    gan = keras.models.Model(inputs=z, outputs=D_G_z)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=0.5,
                                                                               beta_2=0.999))
    return gan, discriminator, generator


def train_gan(discriminator, generator, gan, epochs, batch_size, latent_dim, train_gen):
    """
    Performs the training of the GAN model.
    Args:
        discriminator: discriminator network for the GAN
        generator: generator network for the GAN
        gan: combined network for the GAN
        epochs: Number of epochs to perform training for
        batch_count: Number of batches per epoch
        latent_dim: dimensionality of the input space for the generator
        train_gen: generator for the training dataset.

    Returns:

    """
    d_losses_real = []
    d_losses_fake = []
    g_losses = []
    batch_times = []
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
            y_fake = np.zeros(batch_size)

            # Train discriminator with this batch of samples
            # training has to be seperated between real and fake images for the minibatch discrimination to work
            discriminator.trainable = True
            try:
                d_loss_real = discriminator.train_on_batch(X_real, y_real)
                d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
            except ValueError:
                print(X_real.shape, X_fake.shape, noise.shape)
                continue
            # Train generator with a new batch of generated samples
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])

            # From the generator's perspective, the discriminator should predict
            # ones for all samples
            y_gen = np.ones(batch_size)

            # Freeze the discriminator part
            discriminator.trainable = False

            # train the discriminator exclusively for the first few batches
            if e == 0 and b < 20:
                end_time = time.time()
                batch_times.append(end_time - start_time)
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
                print(
                    '\r%d%% done with epoch number %d, eta: %d seconds, batch time: %f' % (n, e + 1, eta, time_1_batch),
                    end="")
        d_losses_real.append(d_loss_real)
        d_losses_fake.append(d_loss_fake)
        g_losses.append(g_loss)

        if e % 5 == 0:
            noise = np.random.normal(0, 1, size=[100, latent_dim])
            generatedImages = generator.predict(noise)
            generatedImages = generatedImages.reshape(100, 32, 32, 3)
            plotImages(generatedImages,
                       title='Epoch {}'.format(e))  # map pixel values to the [0, 1] range for plotting
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.01)
            saveModels(discriminator, generator, e, 'Upsampling')
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.plot(d_losses_fake)
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
    plotGeneratedImagesPatchCamelyon(epochs, generator)
    return d_losses_real, d_losses_real, g_losses
