import tensorflow as tf
import keras
from keras import layers
from keras import activations
from keras.datasets.mnist import load_data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard


def discriminator():
    model = keras.Sequential()
    # need to go from 28x28x1 to 1 binary classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(1, activation=activations.sigmoid))
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))
    return model


def generator(latent_dim):
    model = keras.Sequential()
    # need to go from a latent dim to a 28x28x1 image
    # 28x28 means a vector of length 784
    model.add(layers.Dense(784, input_shape=(latent_dim,), activation=activations.tanh))
    model.add(layers.Reshape((28, 28, 1)))
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))
    return model

def combined_model(latent_dim):
    g = generator(latent_dim)
    d = discriminator()
    noise = layers.Input(shape=(latent_dim,))
    g_out = g(noise)
    d.trainable = False
    d_out = d(g_out)
    model = keras.models.Model(inputs=noise, outputs=d_out)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))
    return model, g, d

(trainX, trainy), (testX, testy) = load_data()
trainX = (trainX / 255.0).astype(np.float32)
testX = (testX / 255.0).astype(np.float32)
trainy = tf.keras.utils.to_categorical(trainy, 10)
testy = tf.keras.utils.to_categorical(testy, 10)
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

latent_dim = 10
gan, gen, disc = combined_model(latent_dim)

epochs = 10
BATCH_SIZE = 128
batch_count = int(trainX.shape[0] / BATCH_SIZE)
noise = tf.random.normal([1, latent_dim])
gen_losses = []
disc_losses = []
for e in range(epochs):
    for batch in range(batch_count):
        noise = np.random.normal(0, 1, size=[BATCH_SIZE, latent_dim])
        image_batch = trainX[np.random.randint(0, trainX.shape[0], size=BATCH_SIZE)]
        generated = gen(noise)
        X = np.concatenate([image_batch, generated])
        y_dis = np.zeros(2 * BATCH_SIZE)
        y_dis[:BATCH_SIZE] = 0.9
        disc.trainable = True
        disc_loss = disc.train_on_batch(X, y_dis)
        noise = np.random.normal(0, 1, size=[BATCH_SIZE, latent_dim])
        y_gen = np.ones(BATCH_SIZE)
        disc.trainable = False
        gen_loss = gan.train_on_batch(noise, y_gen)
        if batch % 20 == 0:
            print(str(int(batch * 100 / batch_count)) + '% done with epoch number' + str(e))
    gen_losses.append(gen_loss)
    disc_losses.append(disc_loss)
    noise = np.random.normal(0, 1, size=[16, latent_dim])
    predictions = gen(noise)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i], cmap='gray')
        plt.axis('off')
    plt.show()


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
    model = keras.Sequential()
    # need to go from 28x28x1 to 1 binary classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(10, activation=activations.softmax))
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  metrics=['accuracy'])
    return model


transfer = transfer_classifier(disc)
standard = classifier()
model_name_1 = 'transfer'
model_name_2 = "standard"
tensorboard_1 = TensorBoard("logs/" + model_name_1)
tensorboard_2 = TensorBoard("logs/" + model_name_2)
transfer.fit(trainX, trainy, batch_size=128, epochs=10, verbose=1, callbacks=[tensorboard_1])
standard.fit(trainX, trainy, batch_size=128, epochs=10, verbose=1, callbacks=[tensorboard_2])
transfer.summary()
standard.summary()
transfer.evaluate(testX, testy, batch_size=128)
standard.evaluate(testX, testy, batch_size=128)
