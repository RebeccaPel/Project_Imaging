"""
Code to generate a submission to the Kaggle PCAM,
as provided in TU/e BME Project Imaging 2021.

@author: Suzanne Wetstein
"""

'''
TU/e BME Project Imaging 2021
Submission code for Kaggle PCAM
Author: Suzanne Wetstein
'''

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np

import glob
import pandas as pd
from matplotlib.pyplot import imread

import keras
from keras import initializers

import sys
sys.path.append("tools")
from custom_layers import MinibatchDiscrimination

# Input the path to the "Main Project" folder here
base_path = r"C:\Users\justi\Documents\Project_Imaging\Main project"
# Input the path to the test folder of the dataset here
data_path = r'C:\Users\justi\PycharmProjects\pythonProject\test\\'
# load the classifier
init = initializers.get("glorot_uniform")

# open the test set in batches (as it is a very big dataset) and make predictions
test_files = glob.glob(data_path + '*.tif')
model_path = base_path + r"\models"

submission = pd.DataFrame()

file_batch = 5000
max_idx = len(test_files)
print(max_idx)
files = os.listdir(model_path)
files.remove('gan_discriminator_epoch_Upsampling_190.h5')
files.remove('gan_generator_epoch_Upsampling_190.h5')
for file in files:
    model = keras.models.load_model(os.path.join(model_path, file),
                                   custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                   'GlorotUniform': init})
    for idx in range(0, max_idx, file_batch):

        print('Indexes: %i - %i'%(idx, idx+file_batch))

        test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})


        # get the image id 
        test_df['id'] = test_df.path.map(lambda x: x.split(os.sep)[-1].split('.')[0])
        test_df['image'] = test_df['path'].map(imread)
        
        
        K_test = np.stack(test_df['image'].values)
        
        # apply the same preprocessing as during draining
        K_test = K_test.astype('float')/255.0
        
        predictions = model.predict(K_test)
        
        test_df['label'] = predictions
        submission = pd.concat([submission, test_df[['id', 'label']]])


    # save your submission
    submission.head()
    print(len(submission))
    submission.to_csv('submission_'+file[:-4]+'.csv', index = False, header = True)
    submission = pd.DataFrame()


