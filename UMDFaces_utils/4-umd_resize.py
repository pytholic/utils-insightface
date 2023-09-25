import os
import cv2
from tqdm import tqdm

# Define the path to the "imgs_umd" directory
root_dir = '/home/jovyan/haseeb-dataset-3090ti/face-data/faces_emore/imgs_umd'

for person_name in tqdm(os.listdir(root_dir)):
    person_path = os.path.join(root_dir, person_name)
    for img_name in tqdm(os.listdir(person_path)):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (112, 112))
        cv2.imwrite(img_path, img)