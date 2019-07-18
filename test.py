
import os, shutil
import numpy as np
import time
import matplotlib.pyplot as plt
import pdb
import csv
import math
import sys

print('gathering files...')
train_dir = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain/'
train_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
train_pitched_up_dir='/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain+30PercentPitch/'
train_pitched_up_files=[train_pitched_up_dir + x for x in os.listdir(train_pitched_up_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
train_pitched_down_dir='/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain-30PercentPitch/'
train_pitched_down_files=[train_pitched_down_dir + x for x in os.listdir(train_pitched_down_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
train_rate_up_dir='/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain+30PercentSr/'
train_rate_up_files=[train_rate_up_dir + x for x in os.listdir(train_rate_up_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
train_rate_down_dir='/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain-30PercentSr/'
train_rate_down_files=[train_rate_down_dir + x for x in os.listdir(train_rate_down_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
val_dir = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioVal/'
val_files = [val_dir + x for x in os.listdir(val_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
test_dir = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTest/'
test_files = [test_dir + x for x in os.listdir(test_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]

aug_files=[train_pitched_up_files,train_pitched_down_files,train_rate_up_files,train_rate_down_files]

print(len(train_files), 'train files \n', len(val_files), 'validation files \n', len(test_files), 'test files')
label_dir= '/Volumes/Samsung_T5/APP/MXX-git2-/SchulterReproduction/jamendo/labels/' #soon to be changed to 'jamendo/betterlabels'
label_files = [label_dir + x for x in os.listdir(label_dir) if x.endswith('.lab')]
# for testing only
test_folder_dir= '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/testFolder/'
test_folder_files = [test_folder_dir + x for x in os.listdir(test_folder_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]

# pdb.set_trace()

for x in train_pitched_up_files:
	print(x)

