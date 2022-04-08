"""
Code to train the three classifier models:
    - Transfer classifier
    - Regular classifier
    - Densenet classifier

To use: Change the paths and the subsample factor and run the file. Once it has finished running the three models 
will be saved in the models folder.


@author: N. Hartog, J. Kleinveld, A. Masot, R. Pelsser
"""
import keras
from keras import initializers
from keras.callbacks import TensorBoard
import tensorflow as tf

import sys
sys.path.append("tools")

from transfer import transfer_classifier, classifier, pre_trained
from custom_layers import MinibatchDiscrimination
from utils import get_pcam_generators

# set to true if you want to save the trained models.
save = False
# Input the path to the "Main Project" folder here
base_path = r"C:\Users\justi\Documents\Project_Imaging\Main project"
# Input the path to the train+val folder of the dataset here
data_path = r'C:\Users\justi\PycharmProjects\pythonProject\train+val'
image_size = (32, 32)
batch_size = 64
epochs = 20
steps = 36000 // batch_size
val_steps = 16000 // batch_size
# Change this to whatever you're doing, it will automatically change the filenames that it saves to
subsample_factor = 25
model_name_1 = 'transfer'
model_name_2 = 'standard'
model_name_3 = 'efficientnet'

# load the discriminator from the GAN
init = initializers.get("glorot_uniform")
# This should be the path to your saved discriminator file.
discriminator = keras.models.load_model(base_path + '/models/gan_discriminator_epoch_Upsampling_190.h5',
                                        custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                        'GlorotUniform': init})
train_gen, val_gen = get_pcam_generators(data_path,
                                         image_size, batch_size, batch_size)
transfer = transfer_classifier(discriminator)
standard = classifier(discriminator)
pre_trained = pre_trained()
early_stop = keras.callbacks.EarlyStopping(restore_best_weights=True, patience=3)
tensorboard_1 = TensorBoard("logs/" + model_name_1)
tensorboard_2 = TensorBoard("logs/" + model_name_2)
tensorboard_3 = TensorBoard("logs/" + model_name_3)

# do initial training of the transfer model
print("training \n")
print("transfer")
transfer.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=epochs//2, verbose=1,
             callbacks=[tensorboard_1, early_stop], validation_data=val_gen)
# unfreeze the layers of the transfer model and finetune
transfer.trainable = True
transfer.compile(loss="binary_crossentropy",
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999),
                 metrics=['accuracy'])
transfer.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=epochs//2, verbose=1,
             callbacks=[tensorboard_1, early_stop], validation_data=val_gen)

print("standard")
standard.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=epochs//2, verbose=1,
             callbacks=[tensorboard_2, early_stop], validation_data=val_gen)
standard.compile(loss="binary_crossentropy",
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999),
                 metrics=['accuracy'])
standard.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=epochs//2, verbose=1,
             callbacks=[tensorboard_2, early_stop], validation_data=val_gen)

print("pre_trained \n")
pre_trained.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=epochs//2, verbose=1,
                callbacks=[tensorboard_3, early_stop], validation_data=val_gen)
pre_trained.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999),
                    metrics=['accuracy'])
pre_trained.fit(train_gen, batch_size=batch_size, steps_per_epoch=steps, epochs=epochs//2, verbose=1,
                callbacks=[tensorboard_3, early_stop], validation_data=val_gen)

print("evaluating \n")
print("transfer")
transfer.evaluate(val_gen, steps=val_steps, batch_size=batch_size)
print("standard")
standard.evaluate(val_gen, steps=val_steps, batch_size=batch_size)
print("pre_trained")
pre_trained.evaluate(val_gen, steps=val_steps, batch_size=batch_size)
if save:
    keras.models.save_model(transfer, base_path + r"models/transfer_classifier_{}%.h5".format(subsample_factor))
    keras.models.save_model(standard, base_path + r"models/regular_classifer_{}%.h5".format(subsample_factor))
    keras.models.save_model(pre_trained, base_path + r"models/efficientnet_classifer_{}%.h5".format(subsample_factor))