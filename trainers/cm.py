from .base import BaseTrainer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from math import sqrt

class CMTrainer(BaseTrainer):
    """Consistency Model trainer.
    
    Implementation of "Consistency Models" (Song et al., 2023)
    https://arxiv.org/abs/2303.01469
    
    This implements the self-distillation training approach without requiring
    a pre-trained teacher model. Uses EDM-style noise conditioning with learned
    skip connections and distillation between different timesteps.
    """
    def __init__(self, *args, sigma_min=0.002, sigma_max=80.0, 
                 rho=7.0, S_min=0.01, S_max=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Noise schedule parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
        # Distillation parameters
        self.S_min = S_min  # Minimum time ratio for distillation
        self.S_max = S_max  # Maximum time ratio for distillation
    
    def karras_schedule(self, t):
        """Generate sigma schedule following EDM/Karras.
        
        Args:
            t: Timesteps (normalized to [0, 1])
            
        Returns:
            sigma: Noise levels at timesteps t
        """
        return self.sigma_max * (self.sigma_min/self.sigma_max) ** (t/self.nfe)
    
    def sample_time_pairs(self, batch_size):
        """Sample pairs of timesteps for consistency distillation.
        
        Samples (t_init, t_final) pairs where t_init = t_final * S,
        with S uniformly sampled between S_min and S_max.
        
        Args:
            batch_size: Number of time pairs to sample
            
        Returns:
            sigma_init: Initial noise levels
            sigma_final: Final noise levels
        """
        # Sample random initial times
        t_final = torch.rand(batch_size, device=self.device)
        
        # Sample random multipliers between S_min and S_max
        mult = torch.rand(batch_size, device=self.device) * \
               (self.S_max - self.S_min) + self.S_min
        
        # Get paired times
        t_init = t_final * mult
        
        # Convert to sigma values
        sigma_final = self.karras_schedule(t_final * self.nfe)
        sigma_init = self.karras_schedule(t_init * self.nfe)
        
        return sigma_init, sigma_final
    
    def noise(self, x_0, sigma):
        """Forward diffusion process using EDM formulation.
        
        Args:
            x_0: Clean images
            sigma: Noise levels
            
        Returns:
            x_t: Noisy images
            noise: Added noise
        """
        noise = torch.randn_like(x_0)
        x_t = x_0 + noise * sigma[:, None, None, None]
        return x_t, noise
    
    def get_scalings(self, sigma):
        """Get skip connection scalings for given noise levels.
        
        Args:
            sigma: Noise levels
            
        Returns:
            c_skip: Skip connection scaling
            c_out: Output scaling
            c_in: Input scaling
            c_noise: Noise level embedding
        """
        c_skip = 1 / (1 + sigma**2).sqrt()
        c_out = sigma / (1 + sigma**2).sqrt()
        c_in = 1 / (1 + sigma**2).sqrt()
        c_noise = sigma.log() * 0.25
        return c_skip, c_out, c_in, c_noise
    
    def compute_loss(self, x):
        """Training step using consistency distillation"""
        x = x.to(self.device)
        
        # Sample time pairs
        sigma_init, sigma_final = self.sample_time_pairs(x.shape[0])
        
        # Add noise for both timesteps
        x_init, _ = self.noise(x, sigma_init)
        x_final, _ = self.noise(x, sigma_final)
        
        # Get scalings for both timesteps
        c_skip_i, c_out_i, c_in_i, c_noise_i = self.get_scalings(sigma_init)
        c_skip_f, c_out_f, c_in_f, c_noise_f = self.get_scalings(sigma_final)
        
        # Get model predictions
        pred_init = self.model(
            x_init * c_in_i[:, None, None, None],
            c_noise_i
        )
        pred_init = pred_init * c_out_i[:, None, None, None] + \
                   x_init * c_skip_i[:, None, None, None]
        
        pred_final = self.model(
            x_final * c_in_f[:, None, None, None],
            c_noise_f
        )
        pred_final = pred_final * c_out_f[:, None, None, None] + \
                    x_final * c_skip_f[:, None, None, None]
        
        # Compute consistency loss
        loss = F.mse_loss(pred_init.detach(), pred_final)
        
        # Add self-prediction loss
        loss = loss + 0.1 * F.mse_loss(pred_init, x)
        
        return loss
    
    @torch.no_grad()
    def _sample_impl(self, batch_size=16):
        """One-step sampling for Consistency Models.
        
        The key advantage of Consistency Models is their ability
        to generate samples in a single step.
        """
        # Start from noise
        x = torch.randn(batch_size, 1, 28, 28).to(self.device)
        sigma = torch.full((batch_size,), self.sigma_max, device=self.device)
        
        # Get scalings
        c_skip, c_out, c_in, c_noise = self.get_scalings(sigma)
        
        # Single-step prediction
        pred = self.model(
            x * c_in[:, None, None, None],
            c_noise
        )
        x = pred * c_out[:, None, None, None] + x * c_skip[:, None, None, None]
        
        return x.detach().cpu().clamp(0., 1.)