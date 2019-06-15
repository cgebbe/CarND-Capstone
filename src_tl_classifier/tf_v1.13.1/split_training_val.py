import os
import random
import numpy as np
import shutil

if __name__ == '__main__':
    path_dir_train = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/export/splits/train"
    path_dir_valid = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/export/splits/val"
    fraction_valid = 0.20

    labels = [f for f in os.listdir(path_dir_train)
               if os.path.isdir(os.path.join(path_dir_train,f))
               ]
    for label in labels:
        filenames = os.listdir(os.path.join(path_dir_train, label))
        filenames_png = [f for f in filenames if f.endswith('.png')]
        num_files_all = len(filenames_png)

        # pick files to copy
        os.makedirs(os.path.join(path_dir_valid,label), exist_ok=True)
        num_files_val = np.round(num_files_all * fraction_valid).astype(int)
        random.shuffle(filenames_png)
        filenames_val = filenames_png[0:num_files_val]
        for fname in filenames_val:
            shutil.move(os.path.join(path_dir_train, label, fname),
                        os.path.join(path_dir_valid, label, fname),
                        )












