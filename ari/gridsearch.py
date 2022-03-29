# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:06:45 2022

@author: Ari
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from functionsGAN import get_discriminator_histopathology
from utilsGAN import get_pcam_generators, crop
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier

seed = 7
np.random.seed(seed)

# load dataset
train_gen, val_gen = get_pcam_generators(r'D:\Ari\Uni\TUE\8P361', 500)
crop_train_gen = crop(train_gen, 0.333)
crop_val_gen = crop(val_gen, 0.333)


def classifier(learn_rate=0.01, b1=0.9, b2=0.999):
    model = get_discriminator_histopathology()
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=learn_rate, beta_1=b1, beta_2=b2),
                  metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=classifier, verbose=0)

# define the grid search parameters

epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
b1 = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
b2 = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(epochs = epochs, learn_rate=learn_rate, b1=b1, b2=b2)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(crop_train_gen)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))