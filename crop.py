# HOW TO USE:
# 1. OPEN A COMMAND PROMPT (type 'cmd' in your windows search bar, if you're not on windows then idk
# sorry :P)
# 2. USE THE COMMAND: cd path_to_dataset (replace path_to_dataset, it should be to a folder with images. So
# ..\train+val\train\0 for example)
# 3. USE THE COMMAND: python
# 4. COPY THE CODE BELOW AND RUN IT
# this will permanently crop the 96x96 images down to 32x32

from PIL import Image
import os
files = os.listdir()
left = (96-32)/2
top = (96-32)/2
right = (96+32)/2
bottom = (96+32)/2
for file in files:
    im = Image.open(file)
    if im.size == (32, 32):
        continue
    im1 = im.crop((left,top,right,bottom))
    im1.save(file)
