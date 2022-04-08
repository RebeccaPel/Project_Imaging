"""
Code for evaluation of the classifiers. Loads the three trained classifiers and all subsampled variations and calculates,
for each, the accuracy, the ROC curve, AUC and confusion matrix. Also creates visualizations of the layer weights.

@author: N. Hartog, J. Kleinveld, A. Masot, R. Pelsser
"""
import keras
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, auc
from keras.initializers import get
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("tools")
from utils import get_pcam_generators
from custom_layers import MinibatchDiscrimination
from dim_reduction import visualize_first_layer_weights, visualize_layer_weights

# Input the path to the "Main Project" folder here
base_path = r"C:\Users\justi\Documents\Project_Imaging\Main project"
# Input the path to the train+val folder of the dataset here
data_path = r'C:\Users\justi\PycharmProjects\pythonProject\train+val'
image_size = (32, 32)
batch_size = 128
# load the classifier
init = get("glorot_uniform")
train_gen, val_gen = get_pcam_generators(data_path,
                                         image_size, batch_size, batch_size)
# the generator doesn't work properly for evaluating performance with sklearn so we remove the data
# from the generator object
images = []
labels = []
for i in range(16000 // batch_size):
    images.append(val_gen[i][0])
    labels.append(val_gen[i][1])
images = np.array(images)
labels = np.array(labels)
images = np.reshape(images, (16000, 32, 32, 3))
labels = np.reshape(labels, (16000,))

files = os.listdir(base_path + "\models")
files.remove('gan_discriminator_epoch_Upsampling_190.h5')
files.remove('gan_generator_epoch_Upsampling_190.h5')
for i in range(4):
    for j in range(i, 12, 4):
        model = files[j]
        print(model)
        if model[0:12] == "efficientnet":
            efficient = keras.models.load_model(base_path + r'\\models\\' + model,
                                                custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                                'GlorotUniform': init})
        if model[0:7] == "regular":
            standard = keras.models.load_model(base_path + r'\\models\\' + model,
                                                custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                                'GlorotUniform': init})
        if model[0:8] == "transfer":
            transfer = keras.models.load_model(base_path + r'\\models\\' + model,
                                                custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                                'GlorotUniform': init})
    percentage = model.split("_")[2][:-3]

    # generate predicted label probabilities and round them to create label predictions
    y_prob_standard = standard.predict(images)
    y_prob_transfer = transfer.predict(images)
    y_prob_efficient = efficient.predict(images)
    y_pred_standard = np.round(y_prob_standard)
    y_pred_transfer = np.round(y_prob_transfer)
    y_pred_efficient = np.round(y_prob_efficient)

    # calculate model accuracy, confusion matrices and roc curves
    standard_acc = accuracy_score(labels, y_pred_standard)
    transfer_acc = accuracy_score(labels, y_pred_transfer)
    efficient_acc = accuracy_score(labels, y_pred_efficient)
    standard_conf = confusion_matrix(labels, y_pred_standard)
    transfer_conf = confusion_matrix(labels, y_pred_transfer)
    efficient_conf = confusion_matrix(labels, y_pred_efficient)
    standard_fpr, standard_tpr, standard_thresholds = roc_curve(labels, y_prob_standard)
    transfer_fpr, transfer_tpr, transfer_thresholds = roc_curve(labels, y_prob_transfer)
    efficient_fpr, efficient_tpr, efficient_thresholds = roc_curve(labels, y_prob_efficient)
    standard_auc = auc(standard_fpr, standard_tpr)
    transfer_auc = auc(transfer_fpr, transfer_tpr)
    efficient_auc = auc(efficient_fpr, efficient_tpr)
    plt.plot(standard_fpr, standard_tpr, label='Standard {}'.format(percentage), color='orange')
    plt.plot(transfer_fpr, transfer_tpr, label='Transfer {}'.format(percentage), color='b')
    plt.plot(efficient_fpr, efficient_tpr, label='EfficientNetB0 {}'.format(percentage), color='g')
    plt.legend(loc="lower right")
    plt.title('ROC curves, {} of the dataset'.format(percentage))
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.legend(loc="lower right")
    plt.show()

    # print out confusion matrix and accuracy values, and generate the roc plots
    print("standard model {}".format(percentage))
    print("accuracy: {}".format(standard_acc))
    print("AUC: {}".format(standard_auc))
    print("confusion matrix")
    print(standard_conf)
    print("transfer_model {}".format(percentage))
    print("accuracy: {}".format(transfer_acc))
    print("AUC: {}".format(transfer_auc))
    print("confusion matrix")
    print(transfer_conf)
    print("efficientnet model {}".format(percentage))
    print("accuracy: {}".format(efficient_acc))
    print("AUC: {}".format(efficient_auc))
    print("confusion matrix")
    print(efficient_conf)
    
# Generation of the weights for the discriminator and the classifier
init = get("glorot_uniform")
disc = keras.models.load_model(base_path + r'/models/transfer_classifier_100%.h5',
                               custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                   'GlorotUniform': init})
classifier = keras.models.load_model(base_path+ r'/models/regular_classifer_100%.h5',
                                     custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                     'GlorotUniform': init})
visualize_first_layer_weights(disc, 'Transfer')
visualize_first_layer_weights(classifier, 'Standard')
visualize_layer_weights(1, 16, disc, "Transfer NMF second layer")
visualize_layer_weights(1, 16, classifier, "Standard NMF second layer")
visualize_layer_weights(3, 16, disc, "Transfer NMF fourth layer")
visualize_layer_weights(3, 16, classifier, "Standard NMF fourth layer")