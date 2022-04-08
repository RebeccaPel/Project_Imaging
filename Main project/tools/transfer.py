from keras.applications.efficientnet import EfficientNetB0
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Conv2D, LeakyReLU, Flatten, Dropout, Dense, Input
from keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam
def transfer_classifier(transfer_source, lr=0.0005):
    """
    Constructs a classifier with weights transferred from a source model.
    Args:
        transfer_source: model to be used as source

    Returns:
        model: the compiled classification model

    """
    transfer_source.trainable = False
    inputs = transfer_source.inputs
    # take the output from the second to last layer of the full model and
    # replace the last layers with a sigmoid 1-unit dense layer for classification of the PCAM set
    # -5 means the last convolution layer.
    x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), name='convo', padding="same", kernel_regularizer='l1_l2')(
        transfer_source.layers[-7].output)
    x = BatchNormalization()(x)
    x = LeakyReLU(name='leaky')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer='l1_l2')(x)
    model = Model(inputs=inputs, outputs=outputs)
    for i in model.layers[-7:]:
        i.trainable = True
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
    return model


def classifier(model_source, lr=0.0005):
    """
    Constructs a classifier based on a source model, without transferring the weights.
    Args:
        model_source: Source model for the classifier to be based on.
        lr: learning rate for the training

    Returns:
        model: The compiled classification model
    """
    # clone_model() re-initializes the weights.
    model_source = clone_model(model_source)
    inputs = model_source.inputs
    x = BatchNormalization()(model_source.layers[-4].output)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer='l1_l2')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.trainable = True
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
    return model


def pre_trained(lr=0.0005):
    """
    Constructs a pre-trained and compiled mobilenetV2 classifier with imagenet weights.
    Args:
        lr: learning rate for training

    Returns:
        model: the compiled model

    """
    input = Input((32, 32, 3))
    model = EfficientNetB0(input_shape=(32, 32, 3), include_top=False, weights="imagenet")
    batchnorm_indices = [i for i, layer in enumerate(model.layers) if isinstance(layer, BatchNormalization)]
    for i in batchnorm_indices:
        model.layers[i].trainable = True
    model.trainable = False
    model = model(input)
    model = GlobalAveragePooling2D()(model)
    model = Dropout(0.5)(model)
    output = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=input, outputs=output)
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
    return model
