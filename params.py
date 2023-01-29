#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:41:09 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

import logging
import numpy as np
import os
import random
import torch

#%%
'Set Parameter'

'DataSet Path'
target_signals_dir = '../data/data_for_12_leads_reconstructions/normal_ecg'
'Sub_Sampling_Size'
sub_sample = 2
data_check = True
model_prefix = 'Wave_GAN_for_ECG' # name of the model to be saved
use_batchnorm = False
lr_g = 1e-4
"""
you can use with discriminator having a larger learning rate than 
generator instead of using n_critic updates ttur 
https://arxiv.org/abs/1706.08500
"""
lr_d = 1e-4
beta1 = 0.9
beta2 = 0.999
'''
If True: used to linearly deay learning rate untill reaching 0 
at iteration 100,000
'''
decay_lr = False 
'''
In some cases we might try to update the generator with double batch 
size used in the discriminator https://arxiv.org/abs/1706.08500'''
generator_batch_size_factor = 1
'''
update generator every n_critic steps if lr_g = lr_d the n_critic's 
default value is 5 
'''
n_critic = 5
'Gradient penalty regularization factor.'
validate=False
p_coeff = 10
batch_size = 128
noise_latent_dim = 100
'''
model capacity during training can be reduced to 32 for larger window 
length of 2 seconds and 4 seconds
'''
model_capacity_size = 32

"Backup Params"
take_backup = True
output_dir = 'Results'
if not(os.path.isdir(output_dir)):
    os.makedirs(output_dir)

'Signal Parameter'
window_length = 2800 
sampling_rate = 400
num_channels = 12
leads = ["DI", 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

'Logger init'
LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)

'Setting GPU and Seed'
manual_seed = 2019
torch.manual_seed(manual_seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.empty_cache()

elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was "
              "NOT built with MPS enabled.")
    else:
        print("MPS not available because this MacOS version is NOT 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")
