#import packages
import os

#default number of epochs
epochs=24

#default batch size for training
batch_size=32

#noise-level for training
sigma=25.0  #change it according to noise level in your dataset

#path to generate the data
genDataPath='./genData/'

#path to save the genPatches
save_dir='./trainingPatch/'

#path to training data
data='./trainingPatch/img_clean_pats.npy'

#variables required to generate patch
pat_size=30
stride=10
step=0
