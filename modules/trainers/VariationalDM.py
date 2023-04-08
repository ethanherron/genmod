import torch, os
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from einops import rearrange

from torchvision.utils import save_image, make_grid

from modules.networks.Unet import ContextUnet





###############################
# Variational Diffusion utils #
###############################
    
    def gamma(ts, gamma_min=-6, gamma_max=6):
        return gamma_max + (gamma_min - gamma_max) * ts
    
    def sigma2(gamma):
        return nn.Sigmoid(-gamma)
    
    def alpha(gamma):
        return torch.sqrt(1 - sigma2(gamma))
    
    def variance_preserving_map(x, gamma, eps):
        a = alpha(gamma)
        var = sigma2(gamma)
        return a * x + torch.sqrt(var) * eps


#########################
# Variational Diffusion #
#########################  


class VDM(pl.LightningModule):
    def __init__(self,
                n_T=500,
                n_feat=128
                ):
        super(VDM, self).__init__()
        self.nn_model = ContextUnet(in_channels=1, n_feat=n_feat)
        self.n_T = n_T

    def forward(self, x):
        return self.nn_model(x)
    
    def gamma(self, ts, gamma_min=-6, gamma_max=6):
        return gamma_max + (gamma_min - gamma_max) * ts
    
    def sigma2(self, gamma):
        return torch.sigmoid(-gamma)
    
    def alpha(self, gamma):
        return torch.sqrt(1 - self.sigma2(gamma))
    
    def variance_preserving_map(self, x, gamma, eps):
        a = self.alpha(gamma)
        var = self.sigma2(gamma)
        return a * x + torch.sqrt(var) * eps
    
    def loss(self, x):
        # randomly sample timesteps and diffuse x accordingly
        t = torch.randint(1, self.n_T, (x.shape[0],)).to(x).long()  # t ~ Uniform(0, n_T)
        t = torch.ceil(t * self.n_T) / self.n_T
        g_t = self.gamma(t)
        eps = torch.randn_like(x)
        x_t = self.variance_preserving_map(x, g_t, eps)
        # predict noise given noisy images
        eps_hat = self.nn_model(x_t, g_t)
        # compute mse for predicted noise
        loss_recon = torch.mse_loss(eps, eps_hat)
        # loss for finite depth T, i.e. discrete time
        s = t - (1./self.n_T)
        g_s = self.gamma(s)
        loss = .5 * self.n_T * torch.expm1(g_s - g_t) * loss_recon
        return loss
    
    def sample_loop(self, batch):
        x_i = torch.randn_like(batch).to(batch)  # x_T ~ N(0, 1), sample initial noise
        for i in range(self.n_T-1, 0, -1):
            t = (self.n_T - i) / self.n_T
            s = (self.n_T - i - 1) / self.n_T
            g_t = self.gamma(t)
            g_s = self.gamma(s)
            eps = torch.randn_like(x_i).to(x_i)
            # predict noise to remove
            eps_hat = self.nn_model(x_i, g_t)
            # compute terms to denoise x_i w/ eps_hat
            a = torch.sigmoid(g_s)
            b = torch.sigmoid(g_t)
            c = -torch.expm1(g_t - g_s)
            sigma_t = torch.sqrt(self.sigma2(g_t))
            # denoise x_i to x_{i-1}
            x_i = torch.sqrt(a / b) * (x_i - sigma_t * c * eps_hat) + torch.sqrt((1. - a) * c) * eps
        # reverse diffusion to x_0
        g_0 = self.gamma(0.)
        var_0 = self.sigma2(g_0)
        x_0_rescaled = x_i / torch.sqrt(1. - var_0)
        return x_0_rescaled
    
    def training_step(self, batch, batch_idx):
        images, _ = batch
        loss = self.loss(images)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        loss = self.loss(images)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        lr = 1e-4
        opt = torch.optim.Adam(self.nn_model.parameters(), lr=lr)
        return opt
    
    
    
    
    
    
    

