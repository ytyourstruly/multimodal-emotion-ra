# -*- coding: utf-8 -*-
import random
import numpy as np
import os
root = 'd:/Yeskendir_files/RAVDESS_1/RAVDESS'

n_folds=5
indices = list(range(24))

# shuffle them
random.shuffle(indices)

# split into 5 folds (sizes will be 5,5,5,5,4)
folds_raw = np.array_split(indices, 5)
folds_raw = [list(f) for f in folds_raw]

# create final structure: [ [train],[val],[test] ] repeated for 5 folds
folds = []

for i in range(5):
    test_fold = folds_raw[i]
    val_fold = folds_raw[(i + 1) % 5]        # next fold cyclically
    train_fold = []                          # all other folds

    for j in range(5):
        if j != i and j != (i + 1) % 5:
            train_fold.extend(folds_raw[j])

    folds.append([test_fold, val_fold, train_fold])

print("folds = [")
for f in folds:
    print("   ", f, ",")
print("]")
for fold in range(n_folds):
        fold_ids = folds[fold]
        test_ids, val_ids, train_ids = fold_ids
	
        annotation_file = 'annotations_croppad_fold'+str(fold+1)+'.txt'
        # annotation_file = 'annotations.txt'
	
        for i,actor in enumerate(os.listdir(root)):
            for video in os.listdir(os.path.join(root, actor)):
                if not video.endswith('.npy') or 'croppad' not in video:
                    continue
                label = str(int(video.split('-')[2]))
		     
                audio = '03' + video.split('_face')[0][2:] + '_croppad.wav'  
                if i in train_ids:
                   with open(annotation_file, 'a') as f:
                       f.write(os.path.join(root,actor, video) + ';' + os.path.join(root,actor, audio) + ';' + label + ';training' + '\n')
		

                elif i in val_ids:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ label + ';validation' + '\n')
		
                else:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ label + ';testing' + '\n')
		

