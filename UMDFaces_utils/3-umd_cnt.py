import os
import shutil
import cv2

# Define the path to the "imgs_umd" directory
root_dir = '/home/jovyan/haseeb-dataset-3090ti/face-data/faces_emore/imgs_umd'
cnt = 0

# Loop through all directories inside "imgs_umd"
for person_folder in os.listdir(root_dir):
    person_dir = os.path.join(root_dir, person_folder)
    tmp = len(os.listdir(person_dir))
    cnt += tmp

print(cnt)