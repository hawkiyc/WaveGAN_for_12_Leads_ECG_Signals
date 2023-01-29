#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:44:27 2023

@author: hawkiyc
"""

#%%
'Import Library'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from params import *

#%%
'Build the Whole Model'

class Transpose1dLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding=11, upsample=None, output_padding=1,
                 use_batch_norm=False,):
        
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample
        reflection_pad = nn.ConstantPad1d(kernel_size // 2, value=0)
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        conv1d.weight.data.normal_(0.0, 0.02)
        Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, 
                                         kernel_size, stride, padding, 
                                         output_padding)
        batch_norm = nn.BatchNorm1d(out_channels)
        
        if self.upsample:
            operation_list = [reflection_pad, conv1d]
        else:
            operation_list = [Conv1dTrans]

        if use_batch_norm:
            operation_list.append(batch_norm)
            
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x):
        
        if self.upsample:
            
            'recommended by wavgan paper to use nearest upsampling'
            x = nn.functional.interpolate(x, scale_factor=self.upsample, 
                                          mode="nearest")
        
        return self.transpose_ops(x)

class Conv1D(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel_size,
                 alpha=0.2, shift_factor=2, stride=4, padding=11,
                 use_batch_norm=False, drop_prob=0,):
        
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, output_channels, 
                                kernel_size, stride=stride, 
                                padding=padding)
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.phase_shuffle = PhaseShuffle(shift_factor)
        self.alpha = alpha
        self.use_batch_norm = use_batch_norm
        self.use_phase_shuffle = shift_factor == 0
        self.use_drop = drop_prob > 0
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        
        x = self.conv1d(x)
        
        if self.use_batch_norm:
            x = self.batch_norm(x)
        
        x = F.leaky_relu(x, negative_slope=self.alpha)
        
        if self.use_phase_shuffle:
            x = self.phase_shuffle(x)
        
        if self.use_drop:
            x = self.dropout(x)
        
        return x

class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    
    Code from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    """
    
    def __init__(self, shift_factor):
        
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        
        if self.shift_factor == 0:
            return x
        
        'uniform in (L, R)'
        k_list = (torch.Tensor(
            x.shape[0]).random_(0, 2 * self.shift_factor + 1)- 
            self.shift_factor)
        
        k_list = k_list.numpy().astype(int)

        '''Combine sample indices into lists so that less shuffle operations
        need to be performed'''
        k_map = {}
        for idx, k in enumerate(k_list):
            
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        'Make a copy of x for our output'
        x_shuffle = x.clone()

        'Apply shuffle to each sample'
        for k, idxs in k_map.items():
            
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], 
                                        (k, 0), mode="reflect")
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], 
                                        (0, -k), mode="reflect")

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, 
                                                           x.shape)
        
        return x_shuffle

class WaveGANGenerator(nn.Module):
    
    def __init__(self, model_size=64, ngpus=1, num_channels=12, 
                 verbose=False, upsample=True, use_batch_norm=False,):
        
        super(WaveGANGenerator, self).__init__()
        
        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        latent_dim = noise_latent_dim
        self.verbose = verbose
        self.use_batch_norm = use_batch_norm
        self.dim_mul = 16
        self.fc1 = nn.Linear(latent_dim, 4 * model_size * self.dim_mul)
        self.bn1 = nn.BatchNorm1d(num_features=model_size * self.dim_mul)

        stride = 4
        if upsample:
            stride = 1
            upsample = 4

        deconv_layers = [Transpose1dLayer(self.dim_mul * model_size,
                                          (self.dim_mul * model_size) // 2,
                                          25,stride, upsample=upsample,
                                          use_batch_norm=use_batch_norm,),
                         
                         Transpose1dLayer((self.dim_mul * model_size) // 2,
                                          (self.dim_mul * model_size) // 4,
                                          25, stride, upsample=upsample,
                                          use_batch_norm=use_batch_norm,),
                         
                         Transpose1dLayer((self.dim_mul * model_size) // 4,
                                          (self.dim_mul * model_size) // 8,
                                          25, stride, upsample=upsample,
                                          use_batch_norm=use_batch_norm,), 
                         
                         Transpose1dLayer((self.dim_mul * model_size) // 8,
                                          (self.dim_mul * model_size) // 16,
                                          25, stride, upsample=upsample,
                                          use_batch_norm=use_batch_norm,),
                         
                         Transpose1dLayer((self.dim_mul * model_size) // 16,
                                          num_channels, 25, stride,
                                          upsample=upsample,)]
        
        self.deconv_list = nn.ModuleList(deconv_layers)
        
        for m in self.modules():
            
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        
        x = self.fc1(x).view(-1, self.dim_mul * self.model_size, 4)
        
        if self.use_batch_norm:
            x = self.bn1(x)
        
        x = F.relu(x)
        
        if self.verbose:
            print(x.shape)

        for deconv in self.deconv_list[:-1]:
            
            x = F.relu(deconv(x))
            if self.verbose:
                print(x.shape)
                
        output = torch.tanh(self.deconv_list[-1](x))
        output = output[:,:,648:3448]
        return output


class WaveGANDiscriminator(nn.Module):
    
    def __init__(self, model_size=64, ngpus=1, 
                 num_channels=12, shift_factor=2, 
                 alpha=0.2, verbose=False,
                 use_batch_norm=False,):
        
        super(WaveGANDiscriminator, self).__init__()
        
        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.use_batch_norm = use_batch_norm
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        conv_layers = [Conv1D(num_channels, model_size, 25, stride=4,
                              padding=11, use_batch_norm=use_batch_norm,
                              alpha=alpha, shift_factor=shift_factor,),
                       
                       Conv1D(model_size, 2 * model_size, 25, stride=4,
                              padding=11, use_batch_norm=use_batch_norm,
                              alpha=alpha, shift_factor=shift_factor,),
                       
                       Conv1D(2 * model_size, 4 * model_size, 25, stride=4,
                              padding=11, use_batch_norm=use_batch_norm,
                              alpha=alpha, shift_factor=shift_factor,),
                       
                       Conv1D(4 * model_size, 8 * model_size, 25, stride=4,
                              padding=11, use_batch_norm=use_batch_norm,
                              alpha=alpha, shift_factor=shift_factor,),
                       
                       Conv1D(8 * model_size, 16 * model_size, 25, stride=4,
                              padding=11, use_batch_norm=use_batch_norm,
                              alpha=alpha, shift_factor=0,),]
        
        self.fc_input_size = 48 * model_size
        self.conv_layers = nn.ModuleList(conv_layers)
        self.fc1 = nn.Linear(self.fc_input_size, 1)

        for m in self.modules():
            
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        
        for conv in self.conv_layers:
            
            x = conv(x)
            if self.verbose:
                print(x.shape)
                
        x = x.view(-1, self.fc_input_size)
        
        if self.verbose:
            print(x.shape)

        return self.fc1(x)

#%%
'Check Dimension'

if __name__ == "__main__":

    G = WaveGANGenerator(verbose=True, upsample=True, use_batch_norm=True,)
    print('G_Shape_per_Layer')
    out = G(Variable(torch.randn(10, noise_latent_dim)))
    print(out.shape)
    print("==========================")

    D = WaveGANDiscriminator(verbose=True, use_batch_norm=True,)
    print('D_Shape_per_Layer')
    out2 = D(Variable(torch.randn(10, 12, 2800)))
    print(out2.shape)
    print("==========================")