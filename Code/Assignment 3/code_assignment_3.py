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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import model_from_json
from tensorflow.math import confusion_matrix


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
    
     # dataset parameters
     train_path = os.path.join(base_dir, 'train')
     valid_path = os.path.join(base_dir, 'valid')


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
     
     return train_gen, val_gen, datagen


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

     return model


# get the model
model = get_model()


# get the data generators

# Choosing the appropriate path
#path = r'C:\Users\20192157\OneDrive - TU Eindhoven\Documents\Uni\J3-Q3\8P361 Project Imaging'
path = r'C:\Users\justi\PycharmProjects\pythonProject\train+val'

train_gen, val_gen, datagen = get_pcam_generators(path)


# save the model and weights
model_path = r'C:\Users\justi\Documents\Project_Imaging\Assignment models\\'
model_name = 'my_first_cnn_model'
model_filepath = model_path + model_name + '.json'
weights_filepath = model_path + model_name + '_weights.hdf5'

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

# Commented as to not accidentally retrain
# history = model.fit(train_gen, steps_per_epoch=train_steps,
#                     validation_data=val_gen,
#                     validation_steps=val_steps,
#                     epochs=3,
#                     callbacks=callbacks_list)

# hist = history.history # Get relevant information for each epoch

# load the model after it has been trained to avoid long computation times after training
model_json_file = open(model_filepath, 'r')
loaded_model_json = model_json_file.read()
model_json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(weights_filepath)

# the model needs to be recompiled if loaded instead of trained (comment out if training)
model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

# These are the predictions the model makes of the validation images (as per
# the exercise)
images = []
for i in range(500):
    images.append(val_gen[i][0])
images = np.array(images)
images = images.reshape(16000,96,96,3)

val = model.predict(images)

# Now the actual class of the val_gen data is needed:
labels = []
for i in range(500):
    labels.append(val_gen[i][1])
labels = np.array(labels)
labels = labels.reshape((16000,))

## ROC analysis

fpr, tpr, thresholds = roc_curve(labels,val.reshape((val.shape[0],)))
auc_value = auc(fpr,tpr)
plt.plot(fpr,tpr,label = f"AUC first model = {auc_value}")
plt.legend(loc="lower right")
plt.title('ROC curve of the model')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
conf_matrix = confusion_matrix(labels, val.reshape((val.shape[0],)))
print(conf_matrix)

# evaluating the model
score = model.evaluate(val_gen, verbose=0)


## Equivalent model with only convolutional layers

def get_conv_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

      # build the model
      model = Sequential()

      model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
      model.add(MaxPool2D(pool_size = pool_size))

      model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
      model.add(MaxPool2D(pool_size = pool_size))
     
      # The parameters in this layer are adjusted to obtain the same number of 
      # parameters as with the dense layer in the previous model
      model.add(Conv2D(64, (6,6), activation = 'relu', padding = 'same'))
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
conv_model_filepath = model_path + conv_model_name + '.json'
conv_weights_filepath = model_path + conv_model_name + '_weights.hdf5'

conv_model_json = conv_model.to_json() # serialize model to JSON
with open(conv_model_filepath, 'w') as json_file:
     json_file.write(conv_model_json)


# define the model checkpoint and Tensorboard callbacks
conv_checkpoint = ModelCheckpoint(conv_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
conv_tensorboard = TensorBoard(os.path.join('logs', conv_model_name))
conv_callbacks_list = [conv_checkpoint, conv_tensorboard]


# train the model- commented as to not retrain

# conv_history = conv_model.fit(train_gen, steps_per_epoch=train_steps,
#                       validation_data=val_gen,
#                       validation_steps=val_steps,
#                       epochs=3,
#                       callbacks=conv_callbacks_list)


# load the model after it has been trained to avoid long computation times after training
conv_model_json_file = open(conv_model_filepath, 'r')
conv_loaded_model_json = conv_model_json_file.read()
conv_model_json_file.close()
conv_model = model_from_json(conv_loaded_model_json)

# load weights into new model
conv_model.load_weights(conv_weights_filepath)

# the model needs to be recompiled if loaded instead of trained (comment out if training)
conv_model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

# These are the predictions the model makes of the validation images
conv_images = []
for i in range(500):
    conv_images.append(val_gen[i][0])
conv_images = np.array(images)
conv_images = images.reshape(16000,96,96,3)

# These are the predictions the model makes of the validation images
conv_val = conv_model.predict(conv_images)

# Now the actual class of the val_gen data is needed:
labels = []
for i in range(500):
    labels.append(val_gen[i][1])
labels = np.array(labels)
labels = labels.reshape((16000,))

conv_fpr, conv_tpr, conv_thresholds = roc_curve(labels,conv_val)
conv_auc_value = auc(conv_fpr, conv_tpr)
plt.plot(conv_fpr,conv_tpr,label = f"AUC convolutional model = {conv_auc_value}")
plt.legend(loc="lower right")
plt.title('ROC curve comparison for both models')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

conv_conf_matrix = confusion_matrix(labels,conv_val)
print(conv_conf_matrix)

# compare the two models
conv_score = conv_model.evaluate(val_gen, verbose=0)

print('Test loss original model:', score[0])
print('Test loss fully convolutional model:', conv_score[0])
print('Test accuracy original model:', score[1])
print('Test accuracy fully convolutional model:', conv_score[1])