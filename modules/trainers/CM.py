import torch, os
import torch.nn.functional as F
from einops import rearrange
from modules.trainers.base import BaseDiffusionModule

# helper functions
def pad(var):
    if var.shape == ():
        return rearrange(var, ' -> 1 1 1 1')
    else:
        return rearrange(var, 'b -> b 1 1 1')
    
    
    

class CM(BaseDiffusionModule):
    def __init__(self,
                 number_of_timesteps
                ):
        super().__init__()
        
    def forward(self):
        pass
    
    def loss(self, x_0):
        sigmas = pad(self.noise_distribution(x_0))
        x_t = self.forward_diffusion_process(x_0, sigmas)
        # pred
        x_t = x_0 + z * sigma_t
        pred = self.pred_denoise_fn(x_t, t)
        
        # target
        x_tm1 = self.euler_solver(x_t, sigma_t, sigma_tm1, x_0)
        target = self.target_denoise_fn(x_tm1, tm1)
        
        reconstruction_loss = F.mse_loss(pred, target)
        loss = self.loss_weight(sigmas) * reconstruction_loss.mean()
        return loss
    
    def forward_diffusion_process(self, x_t, sigmas):
        eps = torch.randn_like(x_0).to(x_0)
        x_t = x_0 + sigmas * eps
        return x_t
    
    @torch.no_grad()
    def reverse_diffusion_process(self, x_t):
        sigmas = self.sample_schedule()
        for i in range(len(sigmas)-1):
            sigma_t, sigma_tm1 = sigmas[i], sigmas[i+1]
            x_t = self.reverse_diffusion_step(x_t, sigma)
    
    @torch.no_grad()
    def reverse_diffusion_step(self, x_t, sigma_t, sigma_tm1):
        output_tm1 = self.forward(x_t, sigma_t)
        d = (x_t - output_tm1) / sigma_t
        dt = sigma_tm1 - sigma
        x_tm1 = x_t + d * dt
        return x_tm1
        