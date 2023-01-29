#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:10:59 2023

@author: hawkiyc
"""

#%%
"Import Librarys"

import glob
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data

from params import *

#%%
'Generating Latent Space Based on Normal Distribution'

def sample_noise(size):
    
    z = torch.FloatTensor(size, noise_latent_dim).to(device)
    z.data.normal_()
    
    return z

#%%
'Model Utils'

def update_optimizer_lr(optimizer, lr, decay):
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * decay


def gradients_status(model, flag):
    
    for p in model.parameters():
        p.requires_grad = flag


def weights_init(m):
    
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.bias.data.fill_(0)


#%%
'Plotting Utils'

def verify_data(data):
    
    if not (os.path.isdir("check_data")):
        os.makedirs("check_data")
    
    _ = data[:10]
    
    for i, l in enumerate(_):
        
        fig = plt.figure(figsize=(15, 1600))
        
        for s in range(len(l)):
            
            ax2 = fig.add_subplot(12, 1, s + 1,
                                  xticks=[], yticks=[])
            plt.plot(data[i,s].flatten())
            ax2.set_title(f"Lead: {leads[s]}")
        
        plt.savefig(
            f"check_data/check_{i}.png")
        plt.close()

def visualize_ecg(ecg_tensor, iter_indx):
    
    ecg = ecg_tensor.detach().cpu().numpy()
    
    if not (os.path.isdir("interpolated_visualization")):
        os.makedirs("interpolated_visualization")
    
    for i, l in enumerate(ecg):
        
        fig = plt.figure(figsize=(15, 1600))
        
        for s in range(len(l)):
            
            ax2 = fig.add_subplot(12, 1, s + 1,
                                  xticks=[], yticks=[])
            plt.plot(ecg[i,s].flatten())
            ax2.set_title(f"Lead: {leads[s]}")
        
        
        plt.savefig(
            f"interpolated_visualization/epoch_{iter_indx:04d}_{i}.png")
        
        plt.close()

def visualize_loss(loss_1, loss_2, first_legend, second_legend, y_label):
    
    plt.figure(figsize=(10, 5))
    plt.title("{} and {} Loss During Training".format(first_legend, 
                                                      second_legend))
    plt.plot(loss_1, label=first_legend)
    plt.plot(loss_2, label=second_legend)
    plt.xlabel("iterations")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    if not (os.path.isdir("visualization")):
        os.makedirs("visualization")
        
    plt.savefig("visualization/loss.png")
    plt.close()

def latent_space_interpolation(model, iter_indx, n_samples=10,):
    
    z_test = sample_noise(2)
    
    with torch.no_grad():
        
        interpolates = []
        for alpha in np.linspace(0, 1, n_samples):
            
            interpolate_vec = alpha * z_test[0] + ((1 - alpha) * z_test[1])
            interpolates.append(interpolate_vec)

        interpolates = torch.stack(interpolates)
        generated_audio = model(interpolates)
        
    visualize_ecg(generated_audio, iter_indx)

#%%
'File Utils'

def create_batch_reader(data):
    
    batch_idx = np.random.choice(a = np.array(range(len(data))), 
                                 size = batch_size, replace = False)
    batch_ecg = data[batch_idx]
    batch_ecg = torch.Tensor(batch_ecg).to(device)
    
    return batch_ecg

def make_path(output_path):
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    return output_path

def save_samples(epoch_samples, epoch):
    """
    Save G-Output.
    """
    epoch_path = f'epoch_{epoch:04d}'
    sample_dir = make_path(os.path.join(output_dir, epoch_path))
    ecg = epoch_samples
    
    for i, l in enumerate(ecg):
        
        fig = plt.figure(figsize=(15, 1600))
        
        for s in range(len(l)):
            
            ax2 = fig.add_subplot(12, 1, s + 1,
                                  xticks=[], yticks=[])
            plt.plot(ecg[i,s].flatten())
            ax2.set_title(f"Lead: {leads[s]}")
        
        plt.savefig(f'{sample_dir}/{i}.png')
        plt.close()
