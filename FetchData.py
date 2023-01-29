#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 21:54:22 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from params import *
from utils import *

#%%
'Load npz Data'

f_list = [f for f in os.listdir(target_signals_dir) if ".npz" in f]
f_list = f_list[:int(len(f_list)/sub_sample)]

for i in range(len(f_list)):
    
    if i == 0:
        _ = np.load(target_signals_dir+"/"+f_list[i], allow_pickle=True)
        p_id,ecg, = _['p_id'],_['ecg'][:,:,648:3448],
        
    else:
        _ = np.load(target_signals_dir+"/"+f_list[i], allow_pickle=True)
        _p_id,_ecg, = _['p_id'],_['ecg'][:,:,648:3448],
        p_id=np.concatenate((p_id,_p_id))
        ecg=np.concatenate((ecg,_ecg))

N_sample = len(ecg)
scaler = MinMaxScaler(feature_range=(-1, 1))
ecg = np.array([[scaler.fit_transform(np.reshape(l, (-1,1))) 
        for l in idx] for idx in ecg])
ecg = np.squeeze(ecg)

shuffle_index = list(range(len(ecg)))
np.random.shuffle(shuffle_index)
ecg = ecg[shuffle_index]

if validate is False:
    val_ecg = []
else:
    val_ecg = ecg[int(len(ecg)*.9) : ]
    ecg = ecg[:int(len(ecg)*.9)]

#%%
'Check Data'

if data_check:
    verify_data(ecg)


