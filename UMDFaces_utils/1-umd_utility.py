"""
Combine frames and images in one place.
"""

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
                # Loop through image files in the subfolder
                for filename in os.listdir(subfolder_dir):
                    if filename.lower().endswith('.jpg'):
                        src_path = os.path.join(subfolder_dir, filename)
                        dest_path = os.path.join(person_dir, filename)

                        # Copy the image file to the person's main folder
                        shutil.copy(src_path, dest_path)

                        # Optionally, you can remove the image from the subfolder
                        os.remove(src_path)