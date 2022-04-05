"""
Function to permanently crop the central 32x32 patch of the 96x96 images

To use: change dir_path to the folder with the train and valid folders inside.
The code will replace every image to centrally cropped 32x32 versions of the image.

@author: N. Hartog, J. Kleinveld, A. Masot, R. Pelsser
"""


from PIL import Image
import os
dir_path = r"D:\Ari\Uni\TUE\8P361\test"

# paths = [os.path.join(dir_path,r"valid\1"), os.path.join(dir_path,r"valid\0"), 
#         os.path.join(dir_path,r"train\1"), os.path.join(dir_path,r"train\0")]

# for path in paths:
#     files = os.listdir(path)
#     left = (96-32)/2
#     top = (96-32)/2
#     right = (96+32)/2
#     bottom = (96+32)/2
#     for file in files:
#         im = Image.open(os.path.join(path,file))
#         if im.size == (32, 32):
#             continue
#         im1 = im.crop((left,top,right,bottom))
#         im1.save(os.path.join(path,file))

files = os.listdir(dir_path)
left = (96-32)/2
top = (96-32)/2
right = (96+32)/2
bottom = (96+32)/2
for file in files:
    im = Image.open(os.path.join(dir_path,file))
    if im.size == (32, 32):
        continue
    im1 = im.crop((left,top,right,bottom))
    im1.save(os.path.join(dir_path,file))
    
    