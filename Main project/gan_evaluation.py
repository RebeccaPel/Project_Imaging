"""
Code for evaluation of the GAN. Loads the three generator and generates some images,
then plots both real and generated images, and generates the table with the model architecture in the file
model_plot_GAN.
Then calculates the Inception Score and Fréchet Distance for a set of images.

@author: N. Hartog, J. Kleinveld, A. Masot, R. Pelsser
"""

import numpy as np
from numpy import mean, std

import sys
sys.path.append("tools")
from GAN import get_generator_histopathology
from utils import get_pcam_generators, plotImages, calculate_inception_score, scale_images, calculate_fid

from math import floor
from keras.applications.inception_v3 import InceptionV3

from keras.utils.vis_utils import plot_model

# Input the path to the "Main Project" folder here
base_path = r"C:\Users\justi\Documents\Project_Imaging\Main project"
# Input the path to the train+val folder of the dataset here
data_path = r'C:\Users\justi\PycharmProjects\pythonProject\train+val'

latent_dim = 500
image_size = (32, 32)
batch_size = 8
generator = get_generator_histopathology(latent_dim)
generator.load_weights(base_path + r'\models\gan_generator_epoch_Upsampling_190.h5')
train_gen, val_gen = get_pcam_generators(data_path,
                                         image_size, batch_size, batch_size)

# Obtain the plots with real and generated images
for i in range(10):
    noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
    generated = np.asarray(generator(noise))
    plotImages(generated, dim=(2,4), figsize=(10,10), title="Generated Images")
for i in range(10):
    real_images = train_gen[i][0]
    plotImages(real_images, dim=(2,4), figsize=(10,10), title="Real Images")

# Obtain the table with the model architecture
plot_model(generator, to_file='model_plot_GAN.png', show_shapes=True, show_layer_names=True)

batch_size = 1000

# Calculate inception score

model = InceptionV3()
noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
images = generator(noise)
images = np.asarray(images)
images = scale_images(images, (299, 299, 3))
ypred = model.predict(images)
scores = []
n_split = 10
n_part = floor(images.shape[0] / n_split)

for i in range(n_split):
    ix_start, ix_end = i * n_part, i * n_part + n_part
    p_yx = ypred[ix_start:ix_end]
    is_score = calculate_inception_score(p_yx)
    scores.append(is_score)
is_avg, is_std = mean(scores), std(scores)
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))


# Calculate Frechet distance
image_size = (32, 32)
train_gen, val_gen = get_pcam_generators(data_path,
                                         image_size, batch_size, batch_size)
real_images = val_gen[0][0]
real_images = scale_images(real_images, (299, 299, 3))
fake_images = images
fid = calculate_fid(model, real_images, fake_images)
print("inception score: {one}, std: {two}. Distance: {three}".format(one=is_avg, two= is_std, three = fid))

