import keras
from keras import layers
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras import backend as K
from keras.layers import Layer
from keras.layers import InputSpec
from keras import initializers, regularizers, constraints
import tensorflow as tf
import matplotlib.pyplot as plt
from math import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import sklearn.decomposition as decomp


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


def get_order(weights):
    """
    Performs PCA on the input weights and returns an ordering based on the first principal component.
    Args:
        weights:

    Returns:
        pca_ordering: the determined order as an array

    """
    pca = decomp.PCA(n_components=1)
    pca_ordering = np.argsort(pca.fit_transform(
        np.abs(weights)[0, 0])[:, 0])
    return pca_ordering


def plot_images_ordered(images, title, order, dim=[8, 8], figsize=[8, 8]):
    """
    plots images in a given order.
    Args:
        images: images to be plotted
        title: title to be displayed above the plots
        order: the order to plot the images in, array.
        dim: tuple containing the dimensions for the image grid
        figsize: tuple containing width, height of the grid in inches

    Returns:

    """
    plt.figure(figsize=figsize)
    plt.suptitle(title)
    for i, num in enumerate(order):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(images[:, :, :, num])
        plt.title(num)
        plt.axis('off')
    plt.show()


def plot_images(images, title, dim=(5, 5), figsize=(5, 5)):
    """
    plots images in a grid
    Args:
        images: images to be plotted
        title: title to be displayed above the plots
        dim: tuple containing the dimensions for the image grid
        figsize: tuple containing width, height of the grid in inches

    Returns: None
    """
    plt.figure(figsize=figsize)
    plt.suptitle(title)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


def get_weights(model, layer):
    """
    Returns the weights for the specified and the directly following convolutional layers of the
    network (skips over any other layers).
    Args:
        model: Model to extract weights from
        layer: Desired convolutional layer to extract weights from

    Returns:
        layer_weights: weights of the desired convolutional layer
        next_layer_weights: weights of the convolutional layer directly after the desired one. Returns False if the last
        layer is selected

    """
    # model.layers.weights[0] is the slopes and model.layers.weights[1] is the biases.
    # make the weights positve and normalize to a [0, 1] range
    conv_indices = [i for i, layer in enumerate(model.layers) if isinstance(layer, Conv2D)]
    first_layer = conv_indices[layer]
    layer_weights = model.layers[first_layer].weights[0] + model.layers[first_layer].weights[1]
    if conv_indices[-1] == first_layer:
        next_layer_weights = False
    else:
        next_layer = conv_indices[layer + 1]
        next_layer_weights = disc.layers[next_layer].weights[0] + disc.layers[next_layer].weights[1]
    return layer_weights, next_layer_weights


def preprocess_images(images):
    """
    Preprocesses images so they can be properly displayed
    Args:
        images: images to be processed

    Returns:
        images: the preprocessed images

    """
    images = images * 0.5 + 0.5
    images = (images - np.amin(images)) / (np.amax(images) - np.amin(images))
    return images


def visualize_first_layer_weights(model, title):
    """
    Visualizes the weights of the first convolutional layer of a model, ordered by the first PCA component of the
    second convolutional layer.
    Args:
        model: model that the weights should be taken from
    Returns: None

    """
    first_layer_weights, second_layer_weights = get_weights(model, 0)
    images = preprocess_images(first_layer_weights)
    order = get_order(second_layer_weights)
    plot_images_ordered(images, title, order)


def visualize_layer_weights(layer, n_images, model, title):
    """
    Visualizes layer weights for any arbitrary layer by performing dimensionality reduction on them beforehand.
    Dimensionality reduction is done through one-sided NMF.
    Args:
        layer: desired layer to be visualized
        n_images: number of desired kernels to be visualized
        model: model that the weights should be taken from.

    Returns:

    """
    nmf = decomp.NMF(3, init='nndsvd', max_iter=20000)
    w_2, _ = get_weights(model, layer)
    # concatenation hack to make the matrices positive so one sided nmf can be used
    w_2 = np.concatenate([np.maximum(0, w_2), np.maximum(0, -w_2)], axis=2)
    w_2 = np.reshape(w_2, (9, w_2.shape[2], w_2.shape[3]))
    images = np.zeros((n_images, 3, 3, 3))
    root = ceil(np.sqrt(n_images))
    dim = [root, root]
    figsize = (root, root)
    for i in range(n_images):
        w_nmf = nmf.fit_transform(w_2[..., i])
        image = np.reshape(w_nmf, (3, 3, 3))
        images[i] = image
    images = preprocess_images(images)
    plot_images(images, title, dim=dim, figsize=figsize)


# load the models
init = initializers.get("glorot_uniform")
disc = keras.models.load_model(r'C:\Users\justi\Documents\Project_Imaging\Main '
                               r'project\models\transfer_classifier_100%.h5',
                                   custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                   'GlorotUniform': init})
classifier = keras.models.load_model(r'C:\Users\justi\Documents\Project_Imaging\Main '
                                     r'project\models\regular_classifer_100%.h5',
                                     custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                     'GlorotUniform': init})

visualize_first_layer_weights(disc, 'Transfer')
visualize_first_layer_weights(classifier, 'Standard')
visualize_layer_weights(1, 16, disc, "Transfer NMF second layer")
visualize_layer_weights(1, 16, classifier, "Standard NMF second layer")
visualize_layer_weights(3, 16, disc, "Transfer NMF fourth layer")
visualize_layer_weights(3, 16, classifier, "Transfer NMF fourth layer")
