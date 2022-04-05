from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, auc
from transfer import *

image_size = (32, 32)
batch_size = 128
# load the classifier
init = initializers.get("glorot_uniform")
standard = keras.models.load_model(r'C:\Users\justi\Documents\Project_Imaging\Main '
                                   r'project\models\regular_classifer_50%.h5',
                                   custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                   'GlorotUniform': init})
transfer = keras.models.load_model(r'C:\Users\justi\Documents\Project_Imaging\Main project\models\transfer_classifier_50%.h5',
                                   custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                   'GlorotUniform': init})
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject\train+val',
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

# generate predicted label probabilities and round them to create label predictions
y_prob_standard = standard.predict(images)
y_prob_transfer = transfer.predict(images)
y_pred_standard = np.round(y_prob_standard)
y_pred_transfer = np.round(y_prob_transfer)

# calculate model accuracy, confusion matrices and roc curves
standard_acc = accuracy_score(labels, y_pred_standard)
transfer_acc = accuracy_score(labels, y_pred_transfer)
standard_conf = confusion_matrix(labels, y_pred_standard)
transfer_conf = confusion_matrix(labels, y_pred_transfer)
standard_fpr, standard_tpr, standard_thresholds = roc_curve(labels, y_prob_standard)
transfer_fpr, transfer_tpr, transfer_thresholds = roc_curve(labels, y_prob_transfer)
standard_auc = auc(standard_fpr, standard_tpr)
transfer_auc = auc(transfer_fpr, transfer_tpr)
plt.subplot(1, 2, 1)
plt.plot(standard_fpr, standard_tpr, label= "AUC = {}".format(standard_auc))
plt.legend(loc="lower right")
plt.title('ROC curve of the standard model')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.subplot(1, 2, 2)
plt.plot(transfer_fpr, transfer_tpr, label= "AUC = {}".format(transfer_auc))
plt.legend(loc="lower right")
plt.title('ROC curve of the transfer model')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# print out confusion matrix and accuracy values, and generate the roc plots
print("standard model")
print("accuracy: {}".format(standard_acc))
print("confusion matrix")
print(standard_conf)
print("transfer_model")
print("accuracy: {}".format(transfer_acc))
print("confusion matrix")
print(transfer_conf)
