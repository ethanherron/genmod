import torch, os, math
from torch import sqrt
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from einops import rearrange
from tqdm import tqdm

from torchvision.utils import save_image, make_grid

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


class VDM(pl.LightningModule):
    def __init__(self,
                n_T=1000,
                n_feat=128
                ):
        super(VDM, self).__init__()
        self.network = ContextUnet(in_channels=1, n_feat=n_feat)
        self.n_T = n_T

    def forward(self, x_t):
        return self.nn_model(x_t)
    
    # lucid rains logsnr_cosine schedule
    def logsnr(self, t, logsnr_min=-3, logsnr_max=3):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))
    
    def forward_diffusion(self, x_0, alpha, sigma):
        eps = torch.randn_like(x_0).to(x_0)
        x_t = x_0 * alpha + eps * sigma
        return x_t, eps
    
    # this is the v prediction formulation - 
    def loss(self, x_0): 
        # randomly sample timesteps for stochasiticty during training and diffuse x accordingly
        t = torch.zeros((x_0.shape[0],)).to(x_0).float().uniform_(0, 1)
        # compute forward diffusion terms
        log_snr = self.logsnr(t)
        alpha = pad(torch.sqrt(torch.sigmoid(log_snr)))
        sigma = pad(torch.sqrt(torch.sigmoid(-log_snr)))
        x_t, eps = self.forward_diffusion(x_0, alpha, sigma)
        
        # optimize network for v-prediction formulation
        v_hat = self.network(x_t, log_snr)
        v = alpha * eps - sigma * x_0
        return F.mse_loss(v_hat, v)
    
    def reverse_diffusion_process(self, x_t):
        # start from pure gaussian noise and iteratively refine to MNIST-like sample
        steps = torch.linspace(1., 0., self.n_T+1).to(x_t)
        for i in tqdm(range(self.n_T), desc = 'sampling loop time step', total = self.n_T):
            # reverse diffusion step
            if steps[i+1] == 0:
                x_0 = self.reverse_diffusion_step(x_t, steps[i], steps[i+1])
            else:
                x_t = self.reverse_diffusion_step(x_t, steps[i], steps[i+1])
        
        x_0.clamp_(-1., 1.)
        return unnormalize_to_zero_to_one(x_0)
    
    
    def reverse_diffusion_step(self, x_t, t, tm1):
        # single denoising step - 'reverse diffusion step'
        log_snr_t = self.logsnr(t)
        log_snr_tm1 = self.logsnr(tm1)
        squared_alpha_t, squared_alpha_tm1 = torch.sigmoid(log_snr_t), torch.sigmoid(log_snr_tm1)
        squared_sigma_t, squared_sigma_tm1 = torch.sigmoid(-log_snr_t), torch.sigmoid(-log_snr_tm1)

        alpha, sigma, alpha_tm1 = map(sqrt, (squared_alpha_t, squared_sigma_t, squared_alpha_tm1))
        
        v_hat = self.network(x_t, log_snr_t)
        x_delta = alpha * x_t - sigma * v_hat
        
        mu = alpha_tm1 * (x_t * (1 - c) / alpha + c * x_delta)
        
        var = squared_sigma_tm1 * c
        
        x_t = mu + sqrt(var) * torch.randn_like(x_t).to(x_t)
        
        if tm1 == 0:
            return mu
        else:
            return x_t
    
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
    
    
    
    
    
    
    

