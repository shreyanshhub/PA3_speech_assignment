import numpy as np
import os
import pickle
import random

import librosa

import os

def check_dir(rootdir):
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
worker_seed = 0 
np.random.seed(worker_seed)
random.seed(worker_seed)

#%%

target_list = ['clock_tick', 'insects', 'hand_saw', 'crickets', 'water_drops', 'train', 'keyboard_typing', 
               'wind', 'crow', 'airplane', 'sneezing', 'drinking_sipping', 'door_wood_knock', 
               'mouse_click', 'pig', 'chirping_birds', 'sea_waves', 'sheep', 'hen', 'washing_machine', 
               'pouring_water', 'frog', 'crying_baby', 'car_horn', 'cat', 'clock_alarm', 'can_opening', 
               'helicopter', 'engine', 'fireworks', 'rooster', 'crackling_fire', 'cow', 'toilet_flush', 
               'clapping', 'vacuum_cleaner', 'door_wood_creaks', 'footsteps', 'church_bells', 
               'glass_breaking', 'snoring', 'laughing', 'coughing', 'siren', 'thunderstorm', 'rain', 
               'brushing_teeth', 'chainsaw', 'breathing', 'dog']   # for full dataset (50 classes)

audio_path = "./data/ESC-50-master/audio/"   # path to laod raw esc audio
PATH_data = './data/'   # root dir to save the resampled esc audio
meta = np.loadtxt('./data/ESC-50-master/meta/esc50.csv', delimiter=',', dtype='str', skiprows=1)
folds = meta[:, 1]
names = meta[:, 3]
labels = meta[:, 2]

print(len(set(names)))

target_sr = 16000

for fold in [1, 2, 3, 4, 5]:
    print("sampling rate:", target_sr, "fold:", fold)
    train_wav_list = []
    eval_wav_list = []
    eval_labels, training_labels = [], []
    for i in range(len(meta)):
        # loop for target sound classes
        if meta[i, 3] in target_list:     
            y, sr = librosa.load(os.path.join(audio_path, meta[i, 0]), sr=target_sr)
            # resample audio to 16k for feature extraction
            if target_sr < 16000:
                y = librosa.resample(y, orig_sr=target_sr, target_sr=16000)
            label = int(target_list.index(meta[i, 3]))
            y = np.append(y, label)
            # eval set
            if int(meta[i, 1]) == fold:               
                eval_wav_list.append(y)
                eval_labels.append(label)
            # training set
            else:
                train_wav_list.append(y)
                training_labels.append(label)
    
    assert len(set(eval_labels)) == len(target_list)
    assert len(set(training_labels)) == len(target_list)
    
    eval_wav_list = np.asarray(eval_wav_list)
    train_wav_list = np.asarray(train_wav_list)
    
    check_dir(PATH_data + 'split_full_dataset/sr_%d/fold%s/' % (target_sr, fold))
    
    if not os.path.exists(PATH_data + 'split_full_dataset/sr_%d/fold%s/seg_%ds_train_cnn14_inpt_resampled.pkl' % (target_sr, fold, 5)):
        with open(PATH_data + 'split_full_dataset/sr_%d/fold%s/seg_%ds_train_cnn14_inpt_resampled.pkl' % (target_sr, fold, 5), 'wb') as f:
            pickle.dump(train_wav_list, f)
            
    if not os.path.exists(PATH_data + 'split_full_dataset/sr_%d/fold%s/seg_%ds_test_cnn14_inpt_resampled.pkl' % (target_sr, fold, 5)):
        with open(PATH_data + 'split_full_dataset/sr_%d/fold%s/seg_%ds_test_cnn14_inpt_resampled.pkl' % (target_sr, fold, 5), 'wb') as f:
            pickle.dump(eval_wav_list, f)
