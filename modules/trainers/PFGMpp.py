import torch, os, math
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
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
    
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def beta_like(tensor, alpha, beta):
    # get the shape of the input tensor
    shape = tensor.shape

    # create a Beta distribution object with the given alpha and beta
    dist = Beta(alpha, beta)

    # generate a tensor of random samples from the Beta distribution with the same shape as the input tensor
    samples = dist.sample(shape)

    return samples


class PFGMpp(pl.LightningModule):
    def __init__(self,
                n_T=50,
                n_feat=128,
                sigma_min = 0.002,     # min noise level
                sigma_max = 80,        # max noise level
                sigma_data = 0.302,    # standard deviation of data distribution (MNIST)
                rho = 7,               # controls the sampling schedule
                P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
                P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
                S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in paper
                S_tmin = 0.05,
                S_tmax = 50.,
                S_noise = 1.003,
                D = 128,
                N = 28*28                # this is dimension of data i.e., MNIST is 28x28 so N = 784
                ):
        super(PFGMpp, self).__init__()
        self.network = ContextUnet(in_channels=1, n_feat=n_feat)
        self.n_T = n_T

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        
        self.D = D
        self.N = N
    
    def c_in(self, sigmas):
        return 1 * (sigmas ** 2 + self.sigma_data ** 2) ** -0.5
    
    def c_noise(self, sigmas):
        return log(sigmas) * 0.25
    
    def c_skip(self, sigmas):
        return (self.sigma_data ** 2) / (sigmas ** 2 + self.sigma_data ** 2)

    def c_out(self, sigmas):
        return sigmas * self.sigma_data * (self.sigma_data ** 2 + sigmas ** 2) ** -0.5
    
    def forward(self, x_t, sigmas):
        conditioned_input = self.c_in(sigmas) * x_t
        conditioned_noise = self.c_noise(sigmas)
        network_output = self.network(conditioned_input, conditioned_noise)
        x_tm1 = self.c_skip(sigmas) * x_t + self.c_out(sigmas) * network_output
        return x_tm1.clamp(-1., 1.)
    
    
    def noise_distribution(self, x_0):
        return (self.P_mean + self.P_std * torch.randn((x_0.shape[0],)).to(x_0)).exp()
    
    def forward_diffusion(self, x_0, sigmas):
        # sample radii
        r = (sigmas * sqrt(self.D)).to(x_0)
        
        # sample inverse-beta dist
        samples_norm = beta_like(tensor = x_0, alpha = self.N / 2., beta = self.D / 2.).to(x_0).clip(1e-3, 1-1e-3)
        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        
        # sample from p_r(R) by change-of-variables
        samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
        samples_norm = rearrange(samples_norm, 'b c h w -> b (c h w)')
        
        # uniformly sample the angle direction
        gaussian = torch.randn(x_0.shape[0], self.N).to(x_0)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        
        print('unit_gaussian - ', unit_gaussian.shape)
        print('samples_norm - ', samples_norm.shape)
        
        # construct perturbation for x
        perturbation_x = (unit_gaussian * samples_norm)
        n = rearrange(perturbation_x, 'b (c h w) -> b c h w', c=1, h=28, w=28)
        x_t = x_0 + n
        return x_t
    
    def loss_weight(self, sigmas):
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2
    
    def loss(self, x_0): 
        sigmas = pad(self.noise_distribution(x_0))
        
        x_t = self.forward_diffusion(x_0, sigmas)
                
        x_0_hat = self.forward(x_t, sigmas)
        
        reconstruction_loss = F.mse_loss(x_0_hat, x_0)
        loss = (self.loss_weight(sigmas) * reconstruction_loss).mean()
        return loss
        
        
    def sample_schedule(self):
        inv_rho = 1 / self.rho
        
        steps = torch.arange(self.n_T).to(self.device)
        sigmas = (self.sigma_max ** inv_rho + steps / (self.n_T - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
        
        sigmas = F.pad(sigmas, (0,1), value = 0.)
        return sigmas
    
    @torch.no_grad()
    def reverse_diffusion_process(self, x_T):
        sigmas = self.sample_schedule()
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / self.n_T, sqrt(2) - 1),
            0.
        )
        sigmas_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
        
        sigma_T = sigmas[0]
        x_t = sigma_T * torch.randn_like(x_T).to(x_T)
        
        for sigma_t, sigma_tm1, gamma in tqdm(sigmas_gammas, desc = 'sampling time step'):
            x_t = self.reverse_diffusion_step(x_t, sigma_t, sigma_tm1, gamma)
        
        x_0 = x_t  # for consistent variable names across repo, i.e., x_0 == sampled image
        return unnormalize_to_zero_to_one(x_0.clamp_(-1., 1.))
    
    @torch.no_grad()
    def reverse_diffusion_step(self, x_t, sigma_t, sigma_tm1, gamma):
        eps = self.S_noise * torch.randn_like(x_t).to(x_t)
        
        sigma_t_hat = sigma_t + gamma * sigma_t
        x_t_hat = x_t + sqrt(sigma_t_hat ** 2 - sigma_t ** 2) * eps
        
        output_t = self.forward(x_t_hat, sigma_t_hat)
        denoised_over_sigma = (x_t_hat - output_t) / sigma_t_hat
        
        x_tm1 = x_t_hat + (sigma_tm1 - sigma_t_hat) * denoised_over_sigma
        
        # second order correction, unless final timestep
        if sigma_tm1 != 0:
            output_tm1 = self.forward(x_tm1, sigma_tm1)
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
    
    
    
    
    
    
    

