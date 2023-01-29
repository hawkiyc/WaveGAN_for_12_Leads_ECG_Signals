#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:57:25 2023

@author: hawkiyc
"""

#%%
'Import Library'

import math
import numpy as np
import os
import random
from time import sleep
import torch
import torch.optim as optim
from torch.autograd import grad, Variable
from tqdm import tqdm

from FetchData import *
from params import *
from utils import *
from WaveGAN import *

#%%
'Training_Class'

class WaveGan_GP(object):
    def __init__(self, n_epoch):
        
        super(WaveGan_GP, self).__init__()
        
        self.g_cost = []
        self.train_d_cost = []
        self.train_w_distance = []
        self.valid_g_cost = [-1]
        self.valid_reconstruction = []

        self.discriminator = \
            WaveGANDiscriminator(model_size=model_capacity_size,
                                 use_batch_norm=use_batchnorm,
                                 num_channels=num_channels,).to(device)
        self.discriminator.apply(weights_init)

        self.generator = WaveGANGenerator(model_size=model_capacity_size,
                                          use_batch_norm=use_batchnorm,
                                          num_channels=num_channels,
                                          ).to(device)
        self.generator.apply(weights_init)
        
        'opt for G and D'
        self.optimizer_g = optim.Adam(self.generator.parameters(), 
                                      lr=lr_g, betas=(beta1, beta2))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), 
                                      lr=lr_d, betas=(beta1, beta2))
        
        self.validate = validate
        self.n_samples_per_batch = batch_size
        self.n_iterations = n_epoch * math.ceil(N_sample / batch_size)
        self.n_iter_per_epoch = math.ceil(N_sample / batch_size)

    def calculate_discriminator_loss(self, real, generated):
        
        disc_out_gen = self.discriminator(generated)
        disc_out_real = self.discriminator(real)

        alpha = torch.FloatTensor(batch_size, 1, 1).uniform_(0, 1).to(device)
        alpha = alpha.expand(batch_size, real.size(1), real.size(2))

        interpolated = (1 - alpha) * real.data + \
            (alpha) * generated.data[:batch_size]
        interpolated = Variable(interpolated, requires_grad=True)

        'calculate probability of interpolated examples'
        prob_interpolated = self.discriminator(interpolated)
        
        grad_inputs = interpolated
        ones = torch.ones(prob_interpolated.size()).to(device)
        
        gradients = grad(outputs=prob_interpolated, inputs=grad_inputs,
                         grad_outputs=ones, create_graph=True,
                         retain_graph=True, only_inputs=True,)[0]
        
        "calculate gradient penalty"
        grad_penalty = (p_coeff * ((
            gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean())
        
        assert not (torch.isnan(grad_penalty))
        assert not (torch.isnan(disc_out_gen.mean()))
        assert not (torch.isnan(disc_out_real.mean()))
        
        cost_wd = disc_out_gen.mean() - disc_out_real.mean()
        cost = cost_wd + grad_penalty
        
        return cost, cost_wd

    def apply_zero_grad(self):
        
        self.generator.zero_grad()
        self.optimizer_g.zero_grad()

        self.discriminator.zero_grad()
        self.optimizer_d.zero_grad()

    def enable_disc_disable_gen(self):
        
        gradients_status(self.discriminator, True)
        gradients_status(self.generator, False)
        
    def enable_gen_disable_disc(self):
        
        gradients_status(self.discriminator, False)
        gradients_status(self.generator, True)

    def disable_all(self):
        
        gradients_status(self.discriminator, False)
        gradients_status(self.generator, False)

    def train(self):
        
        progress_bar = tqdm(total=
                            self.n_iterations // self.n_iter_per_epoch)
        
        'For Forged ECG while Saving Results'
        fixed_noise = sample_noise(batch_size).to(device)  

        gan_model_name = "{}.tar".format(model_prefix)

        first_iter = 0
        
        if take_backup and os.path.isfile(gan_model_name):
            
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                checkpoint = torch.load(gan_model_name)
            else:
                checkpoint = torch.load(gan_model_name, map_location="cpu")
            
            self.generator.load_state_dict(checkpoint["generator"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
            self.train_d_cost = checkpoint["train_d_cost"]
            self.train_w_distance = checkpoint["train_w_distance"]
            self.valid_g_cost = checkpoint["valid_g_cost"]
            self.g_cost = checkpoint["g_cost"]

            first_iter = checkpoint["n_iterations"] + 1
            
            for i in range(int(first_iter/self.n_iter_per_epoch)):
                sleep(0.01)
                progress_bar.update()
                
            self.generator.eval()
            
            with torch.no_grad():
                fake = self.generator(fixed_noise).detach().cpu().numpy()
            save_samples(fake, int(first_iter/self.n_iter_per_epoch))
            
        self.generator.train()
        self.discriminator.train()
        
        for iter_indx in range(first_iter, self.n_iterations):
            
            self.enable_disc_disable_gen()
            
            for _ in range(n_critic):
                
                real_signal = create_batch_reader(ecg)

                'Creat Forged ECG'
                noise = sample_noise(batch_size * generator_batch_size_factor)
                generated = self.generator(noise)
                
                'Calculating discriminator loss and updating discriminator'
                
                self.apply_zero_grad()
                disc_cost, disc_wd = self.calculate_discriminator_loss(
                    real_signal.data, generated.data)
                
                assert not (torch.isnan(disc_cost))
                
                disc_cost.backward()
                self.optimizer_d.step()

            if self.validate and (iter_indx+1) % self.n_iter_per_epoch == 0:
                
                self.disable_all()
                
                val_data = create_batch_reader(val_ecg)
                val_real = val_data
                
                with torch.no_grad():
                    
                    val_discriminator_output = self.discriminator(val_real)
                    val_generator_cost = val_discriminator_output.mean()
                    self.valid_g_cost.append(val_generator_cost.item())
            
            'Update G network every n_critic steps'
            self.apply_zero_grad()
            self.enable_gen_disable_disc()
            noise = sample_noise(batch_size * generator_batch_size_factor)
            generated = self.generator(noise)
            discriminator_output_fake = self.discriminator(generated)
            generator_cost = -discriminator_output_fake.mean()
            generator_cost.backward()
            self.optimizer_g.step()
            self.disable_all()

            if (iter_indx+1) % self.n_iter_per_epoch == 0:
                
                self.g_cost.append(generator_cost.item() * -1)
                self.train_d_cost.append(disc_cost.item())
                self.train_w_distance.append(disc_wd.item() * -1)

                progress_updates = {
                    "Loss_D WD": str(self.train_w_distance[-1]),
                    "Loss_G": str(self.g_cost[-1]),
                    "Val_G": str(self.valid_g_cost[-1]),}
                
                progress_bar.set_postfix(progress_updates)
                progress_bar.update()
            
            'lr decay'
            if decay_lr:
                
                decay = max(0.0, 1.0 - (iter_indx * 1.0 / self.n_iterations))
                
                'update the learning rate'
                update_optimizer_lr(self.optimizer_d, lr_d, decay)
                update_optimizer_lr(self.optimizer_g, lr_g, decay)

            if (iter_indx+1) % self.n_iter_per_epoch == 0:
                
                with torch.no_grad():
                    latent_space_interpolation(self.generator, 
                                               int(
                                                   (1+iter_indx)/
                                                   self.n_iter_per_epoch), 
                                               n_samples=2,)
                    fake = self.generator(fixed_noise).detach().cpu().numpy()
                    
                save_samples(fake, int((1+iter_indx)/self.n_iter_per_epoch))

            if take_backup and (iter_indx+1) % self.n_iter_per_epoch == 0:
                saving_dict = {
                    "generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict(),
                    "n_iterations": iter_indx,
                    "optimizer_d": self.optimizer_d.state_dict(),
                    "optimizer_g": self.optimizer_g.state_dict(),
                    "train_d_cost": self.train_d_cost,
                    "train_w_distance": self.train_w_distance,
                    "valid_g_cost": self.valid_g_cost,
                    "g_cost": self.g_cost,}
                
                torch.save(saving_dict, gan_model_name)
                
        self.generator.eval()

#%%
'Training the Model'

if __name__ == "__main__":
    
    wave_gan = WaveGan_GP(200)
    wave_gan.train()
    visualize_loss(wave_gan.g_cost, wave_gan.valid_g_cost, 
                   "Train", "Val", "Negative Critic Loss")
