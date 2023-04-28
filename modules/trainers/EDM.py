import torch, os, math
from torch import sqrt
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from einops import rearrange
from tqdm import tqdm

from modules.networks.Unet import ContextUnet


# helper functions

# pad alpha, sigma terms from [b] -> [b 1 1 1]
def pad(var):
    if var.shape == ():
        return rearrange(var, ' -> 1 1 1 1')
    else:
        return rearrange(var, 'b -> b 1 1 1')

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class EDM(pl.LightningModule):
    def __init__(self,
                n_T=1000,
                n_feat=128
                ):
        super(EDM, self).__init__()
        self.network = ContextUnet(in_channels=1, n_feat=n_feat)
        self.n_T = n_T

    def forward(self, x_t):
        return self.nn_model(x_t)
    
    
    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,)).to(self.device)).exp()
    
    
    def forward_diffusion(self, x_0, sigmas):
        eps = torch.randn_like(x_0).to(x_0)
        x_t = x_0 + sigmas * eps
        return x_t
    
    
    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2
    
    
    def loss(self, x_0): 
        sigmas = pad(self.noise_distribution(x_0.shape[0]))
        x_t = self.forward_diffusion(x_0, sigmas)
        
#         self_cond = None
        
#         if self.self_condition and random() < 0.5:
#             with torch.no_grad():
#                 self_cond = self.preconditioned_net(x_t, sigmas)
#                 self_cond.detach_()
                
        x_tm1 = self.preconditioned_net(x_t, sigmas)#, self_cond)
        
        reconstruction_loss = F.mse_loss(x_tm1, images)
        loss = (self.loss_weight(sigmas) * reconstruction_loss).mean()
        return loss
        
    def sample_schedule(self):
        inv_rho = 1 / self.rho
        
        steps = torch.arange(self.n_T).to(self.device)
        sigmas = (self.sigma_max ** inv_rho + steps / (self.n_T - 1) 
                  * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
        
        sigmas = F.pad(sigmas, (0,1), value = 0.)
        return sigmas
    
    @torch.no_grad()
    def reverse_diffusion_process(self, x_T):
        sigmas = self.sample_schedule()
        gammas = torch.where((sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
                             min(self.S_churn / self.n_T, sqrt(2) - 1),
                             0.
                            )
        sigmas_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
        
        sigma_T = sigmas[0]
        x_t = init_sigma * torch.randn_like(x_T).to(x_T)
        for sigma_t, sigma_tm1, gamma in tqdm(sigmas_gammas, desc = 'sampling time step'):
            x_t = self.reverse_diffusion_step(x_t, sigma_t, sigma_tm1, gamma)
            
        return unnormalize_to_zero_to_one(x_0.clamp_(-1., 1.))
    
    def reverse_diffusion_step(self, x_t, sigma_t, sigma_tm1, gamma):
        sigma_t, sigma_tm1, gamma = map(lambda t: t.item(), (sigma_t, sigma_tm1, gamma))
        
        eps = self.S_noise * torch.randn_like(x_t).to(x_t)
        
        sigma_t_hat = sigma_t + gamma * sigma_t
        x_t_hat = x_t + sqrt(sigma_t_hat ** 2 - sigma ** 2) * eps
        
        output_t = self.preconditioned_network(x_t_hat, sigma_t_hat)
        denoised_over_sigma = (x_t_hat - ouput_t) / sigma_t_hat
        
        x_tm1 = x_t_hat + (sigma_tm1 - sigma_t_hat) * denoised_over_sigma
        
        # second order correction, unless final timestep
        if sigma_tm1 != 0:
            output_tm1 = self.network(x_tm1, sigma_tm1)
            denoised_prime_over_sigma = (x_tm1 - output_tm1) / sigma_tm1
            x_tm1 = x_t_hat + 0.5 * (sigma_tm1 - sigma_t_hat) * (denoised_over_sigma + denoised_prime_over_sigma)
            
        return x_tm1
        
        
    
    def training_step(self, batch, batch_idx):
        images, _ = batch
        images = normalize_to_neg_one_to_one(images)
        loss = self.loss(images)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        images = normalize_to_neg_one_to_one(images)
        loss = self.loss(images)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        lr = 1e-4
        opt = torch.optim.Adam(self.network.parameters(), lr=lr)
        return opt
    
    
    
    
    
    
    

