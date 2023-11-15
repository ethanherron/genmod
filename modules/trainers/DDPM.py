import torch, os
import torch.nn.functional as F
import lightning as pl
from einops import rearrange

from torchvision.utils import save_image, make_grid

from modules.networks.UnetViT import UViT
from modules.networks.Unet import ContextUnet
from modules.utils import linear_beta_schedule

from modules.trainers.base import BaseDiffusionModule


# helper functions
def pad(var):
    if var.shape == ():
        return rearrange(var, ' -> 1 1 1 1')
    else:
        return rearrange(var, 'b -> b 1 1 1')


class DDPM(BaseDiffusionModule):
    def __init__(self, 
                 number_of_timesteps=1000
                 ):
        super().__init__(number_of_timesteps)
        
        self.ddpm_schedules = self.register_ddpm_schedules()
        for k, v in self.ddpm_schedules.items():
            self.register_buffer(k, v)

    def register_ddpm_schedules(self):
        """
        Returns pre-computed schedules for DDPM sampling, training process.
        """
        
        beta_t = linear_beta_schedule(self.n_T)
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def forward(self, x, t):
        return self.nn_model(x_t, t / self.n_T)
    
    def loss(self, x):
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x).long()  # t ~ Uniform(0, n_T)
        x_t = self.forward_diffusion_process(x, _ts, noise)
        return F.mse_loss(noise, self.nn_model(x_t, _ts / self.n_T))

    def forward_diffusion_process(self, x, t, noise):
        x_t = pad(self.sqrtab[t]) * x + pad(self.sqrtmab[t]) * noise
        return x_t
    
    def reverse_diffusion_step(self, x_t, t, z, i):
        eps = self.nn_model(x_t, t)
        
        x_t_minus_1 = (
                self.oneover_sqrta[i] * (x_t - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        # x_t_minus_1 = x_t + h * dx/dt
        return x_t_minus_1

    def reverse_diffusion_process(self, x_t):
        for i in range(self.n_T-1, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(x_t)
            t_is = t_is.repeat(x_t.size(0),1,1,1)
            z = torch.randn_like(x_t).to(x_t) if i > 1 else 0

            x_t = self.reverse_diffusion_step(x_t, t_is, z, i)
            
        return x_t