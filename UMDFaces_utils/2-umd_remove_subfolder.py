import os
import shutil

# Define the path to the "imgs_umd" directory
root_dir = '/home/jovyan/haseeb-dataset-3090ti/face-data/faces_emore/imgs_umd'

# Loop through all directories inside "imgs_umd"
for person_folder in os.listdir(root_dir):
    person_dir = os.path.join(root_dir, person_folder)

    # Check if the item in "imgs_umd" is a directory
    if os.path.isdir(person_dir):
        # Loop through subdirectories (0001, 0002, etc.) inside the person's folder
        for subfolder in os.listdir(person_dir):
            subfolder_dir = os.path.join(person_dir, subfolder)
            # Check if the subfolder is a directory
            if os.path.isdir(subfolder_dir):
                if len(os.listdir(subfolder_dir)) == 0:
                    os.rmdir(subfolder_dir)