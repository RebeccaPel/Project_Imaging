import sys
import numpy as np
sys.path.append("")
from GAN import get_generator_histopathology
from utils import get_pcam_generators, plotImages
latent_dim = 500
image_size = (32, 32)
batch_size = 8
generator = get_generator_histopathology(latent_dim)
generator.load_weights(r'C:\Users\justi\Documents\Project_Imaging\Main project\gan_generator_epoch_Upsampling_190.h5')
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject\train+val',
                                         image_size, batch_size, batch_size)
for i in range(10):
    noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
    generated = np.asarray(generator(noise))
    plotImages(generated, dim=(2,4), figsize=(10,10), title="Generated Images")
for i in range(10):
    real_images = train_gen[i][0]
    plotImages(real_images, dim=(2,4), figsize=(10,10), title="Real Images")
