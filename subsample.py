# To run, change the paths and the subsample factor and run the file!
# This will create directories as necessary containing your subsampled data.
import os
import random
import shutil
# change these as necessary
original_path = r"D:\Ari\Uni\TUE\8P361\train+val\train"
subsampled_path = r"D:\Ari\Uni\TUE\8P361\train+val_sub_75\train"
# change this to whatever you want
subsample_factor = 0.75
try:
    os.makedirs(subsampled_path)
    os.mkdir(subsampled_path + r"\0")
    os.mkdir(subsampled_path + r"\1")
except OSError:
    print("Directory already exists, please choose a different directory for your subsampled data.")
    raise OSError
files_0 = os.listdir(original_path + r'\0')
files_1 = os.listdir(original_path + r'\1')
sample_0 = random.sample(files_0, int(subsample_factor*len(files_0)))
sample_1 = random.sample(files_1, int(subsample_factor*len(files_1)))
for file in sample_0:
    shutil.copy2(original_path + r'\0\\' + file, subsampled_path + r'\0')
for file in sample_1:
    shutil.copy2(original_path + r'\1\\' + file, subsampled_path + r'\1')





