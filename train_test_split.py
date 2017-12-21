import os
import re
import numpy as np
import shutil

files = os.listdir()
images = [img.split(".")[0] for img in files if img[-3:] == 'jpg']
random_images = np.random.permutation(images)
split_loc = int(np.floor(len(images) * .9))
train_file_pairs = [(img+".jpg",img+".xml") for img in random_images[:split_loc]]
test_file_pairs = [(img+".jpg",img+".xml") for img in random_images[split_loc:]]

image_directory = os.getcwd()
train_directory = os.path.join(image_directory,'train')
test_directory = os.path.join(image_directory,'test')

for pair in train_file_pairs:
    for file in pair:
        shutil.copy(image_directory+"/"+file,train_directory)
for pair in test_file_pairs:
    for file in pair:
        shutil.copy(image_directory+"/"+file,test_directory)
