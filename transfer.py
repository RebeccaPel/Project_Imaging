import gzip
import os
import pickle

import keras
import tensorflow as tf
from keras import layers, initializers, regularizers, constraints
from keras.layers import Layer
from keras import backend as K
from keras.layers import InputSpec
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 32
batch_size = 256


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

def get_pcam_generators(base_dir, train_batch_size=batch_size, val_batch_size=batch_size):
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
                                          batch_size=train_batch_size,
                                          class_mode='binary')

    return train_gen, val_gen


def loadPatchCamelyon(path):
    f = gzip.open(path, 'rb')
    train_set = pickle.load(f, encoding='latin1')
    f.close()
    return train_set


init = initializers.get("glorot_uniform")
discriminator = keras.models.load_model('gan_discriminator_epoch_lower_lr145.h5',
                               custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                               'GlorotUniform': init})
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input = Input(input_shape)
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject')
def crop(gen, percentage):
    for x in gen:
        yield tf.image.central_crop(x[0], percentage), x[1]

def transfer_classifier(transfer_source):
    transfer_source.trainable = False
    inputs = transfer_source.inputs
    # take the output from the second to last layer of the full model and
    # replace the last layers with a sigmoid 1-unit dense layer for classification of the PCAM set
    # -5 means the last convolution layer.
    x = Conv2D(256, kernel_size=(3,3), strides= (2,2), name = 'convo', padding = "same")(transfer_source.layers[-7].output)
    x = LeakyReLU(name='leaky')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    for i in model.layers[-7:]:
        i.trainable = True
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    model.summary()
    return model


def classifier(transfer_source):
    transfer_source = keras.models.clone_model(transfer_source)
    inputs = transfer_source.inputs
    x = Dropout(0.2)(transfer_source.layers[-4].output)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.trainable = True
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    model.summary()
    return model


steps = 144000 // batch_size // 20
transfer = transfer_classifier(discriminator)
standard = classifier(discriminator)
model_name_1 = 'transfer'
model_name_2 = "standard"
tensorboard_1 = TensorBoard("logs/" + model_name_1)
tensorboard_2 = TensorBoard("logs/" + model_name_2)
# do initial training of the transfer model
#transfer.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=20, verbose=1, callbacks=[tensorboard_1])
# unfreeze the layers of the transfer model and finetune
#transfer.trainable = True
#transfer.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999),
#              metrics=['accuracy'])
#transfer.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=15, verbose=1, callbacks=[tensorboard_1])
standard.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=20, verbose=1, callbacks=[tensorboard_2])
standard.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999),
                 metrics=['accuracy'])
standard.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=15, verbose=1, callbacks=[tensorboard_2])
#transfer.evaluate(val_gen, steps=steps, batch_size=batch_size)
standard.evaluate(val_gen, steps=steps, batch_size=batch_size)
#keras.models.save_model(transfer, r"transfer_classifier.h5")
keras.models.save_model(standard, r"regular_classifer.h5")