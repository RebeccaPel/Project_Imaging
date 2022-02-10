
import cv2
from matplotlib import pyplot as plt
import glob
import random as rand

# Because this python file is located in the GitHub repository, a path needs to be defined to the location of the images
path_train = r"C:\Users\20192157\OneDrive - TU Eindhoven\Documents\Uni\J3-Q3\8P361 Project Imaging\train"

class_1 = "\\1\\"
class_0 = "\\0\\"

# Use glob to create all pathways to the images
images_class_1 = glob.glob(path_train + class_1 + '*.jpg', recursive=False)
images_class_0 = glob.glob(path_train + class_0 + '*.jpg', recursive=False)
    
def display_random_images(number_images,images,title):
    '''
    This function displays multiple random images in a 2-by-X frame.
    
    :param number_images: the number of random images which should be displayed
    :type number_images: int
    :param images: A list of all the pathways where the images are located
    :type images: list
    :param title: What main title should be given to the images
    :type title: str
    '''
    
    max = len(images) # Class 1 same lenght as class 0
    rand_list = []
    for i in range(number_images):
        rand_list.append(rand.randint(0,max))
    
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(title)  
    
    columns = 2
    rows = number_images // 2
    if rows %2 != 0:
        rows += 1  

    for i in range(number_images):
    
        fig.add_subplot(rows, columns, i+1)
    
        image = cv2.imread(images_class_1[rand_list[i]])
    
        plt.imshow(image)
        plt.axis('off')
        plt.title(str(i+1))
 
# Calling the function:
display_random_images(7,images_class_1,"Class 1")
display_random_images(6,images_class_0,"Class 0")   