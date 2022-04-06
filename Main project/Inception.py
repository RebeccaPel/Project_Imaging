from numpy import expand_dims, log, mean, exp, std, cov, trace, iscomplexobj
from scipy.linalg import sqrtm
import sys
sys.path.append("tools")
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.transform import resize
from GAN import get_generator_histopathology
from utils import get_pcam_generators
import numpy as np
import keras
from keras.layers import UpSampling2D
import matplotlib.pyplot as plt
import math

# Sources used for code:
# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/


def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

batch_size = 1000
latent_dim = 500

# calculate inception score
model = InceptionV3()
noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
generator = get_generator_histopathology(latent_dim)
generator.load_weights(r'C:\Users\justi\Documents\Project_Imaging\Main project\gan_generator_epoch_Upsampling_190.h5')
images = generator(noise)
images = np.asarray(images)
images = scale_images(images, (299, 299, 3))
plt.imshow(images[0])
plt.show()
ypred = model.predict(images)
scores = []
eps = 1e-16
n_split = 10
n_part = math.floor(images.shape[0] / n_split)
for i in range(n_split):
    ix_start, ix_end = i * n_part, i * n_part + n_part
    p_yx = ypred[ix_start:ix_end]
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    sum_kl_d = kl_d.sum(axis=1)
    avg_kl_d = mean(sum_kl_d)
    is_score = exp(avg_kl_d)
    scores.append(is_score)
is_avg, is_std = mean(scores), std(scores)
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))



# calculate Frechet distance
image_size = (32, 32)
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject\train+val',
                                         image_size, batch_size, batch_size)
real_images = val_gen[0][0]
real_images = scale_images(real_images, (299, 299, 3))
fake_images = images
fid = calculate_fid(model, real_images, fake_images)
print("inception score: {one}, std: {two}. Distance: {three}".format(one=is_avg, two= is_std, three = fid))