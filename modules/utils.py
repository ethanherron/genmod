import torch, math
import numpy as np


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    
    :param timesteps: the total number of timesteps in the diffusion process.
    """
    scale = 1000 / timesteps
    beta_start = scale * 1e-4
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
        
def _update_ema(self):
    for rate, params in zip(self.ema_rate, self.ema_params):
        update_ema(params, self.mp_trainer.master_params, rate=rate)