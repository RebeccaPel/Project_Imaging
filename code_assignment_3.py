'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train')#'train+val', 'train')
     valid_path = os.path.join(base_dir, 'valid')#'train+val', 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')
     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')
     
     return train_gen, val_gen


def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))

     
     # print summary of the model
     model.summary()

     # compile the model
     model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
     

     # compile the model
     model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model


# get the model
model = get_model()


# get the data generators
#train_gen, val_gen = get_pcam_generators(r'C:\Users\20192157\OneDrive - TU Eindhoven\Documents\Uni\J3-Q3\8P361 Project Imaging')
train_gen, val_gen = get_pcam_generators(r'D:\Ari\Uni\TUE\8P361')



# save the model and weights
model_name = 'my_first_cnn_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks_list)


## ROC analysis

# These are the predictions the model makes of the validation images (as per
# the exercise)
val = model.predict(val_gen)

# Now the actual class of the val_gen data is needed:
labels = val_gen.labels

fpr, tpr, thresholds = roc_curve(labels,val)
auc_value = auc(fpr,tpr)
plt.plot(fpr,tpr,label = f"AUC = {auc_value}")
plt.legend(loc="lower right")
plt.show()
score = model.evaluate(val_gen, verbose=0)

# The code has run and has been saved in the GitHub and can be retrieved:



## Exercise 2

def get_conv_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     
     model.add(Conv2D(64, (6,6), activation = 'relu', padding = 'same')) # this was calculated to give the same parameter number (in total)
     model.add(MaxPool2D(pool_size = pool_size))
     
     model.add(Flatten())
     model.add(Dense(1, activation = 'sigmoid'))

     # print summary of the model
     model.summary()

     # compile the model
     model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
     
     

     return model

# get the model
conv_model = get_conv_model()


# save the model and weights
conv_model_name = 'my_conv_cnn_model'
conv_model_filepath = model_name + '.json'
conv_weights_filepath = conv_model_name + '_weights.hdf5'

conv_model_json = conv_model.to_json() # serialize model to JSON
with open(conv_model_filepath, 'w') as json_file:
    json_file.write(conv_model_json)


# define the model checkpoint and Tensorboard callbacks
conv_checkpoint = ModelCheckpoint(conv_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
conv_tensorboard = TensorBoard(os.path.join('logs', model_name))
conv_callbacks_list = [conv_checkpoint, conv_tensorboard]


# train the model


conv_history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=conv_callbacks_list)


# compare the two models

# These are the predictions the model makes of the validation images (as per
# the exercise)
val = model.predict(val_gen)

# Now the actual class of the val_gen data is needed:
labels = val_gen.labels#class_indices#.keys()

fpr, tpr, thresholds = roc_curve(labels,val)
plt.plot(fpr,tpr)

conv_score = conv_history.evaluate(val_gen, labels, verbose=0)

print('Test loss original model:', score[0])
print('Test loss fully convolutional model:', conv_score[0])
print('Test accuracy original model:', score[1])
print('Test accuracy fully convolutional model:', conv_score[1])