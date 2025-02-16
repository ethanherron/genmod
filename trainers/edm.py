from .base import BaseTrainer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from math import sqrt

class EDMTrainer(BaseTrainer):
    """Elucidated Diffusion Model trainer.
    
    Implementation of "Elucidated Diffusion" (Karras et al., 2022)
    https://arxiv.org/abs/2206.00364
    
    Uses continuous sigma parameterization and improved network conditioning
    for better sample quality and training stability.
    """
    def __init__(self, *args, sigma_min=0.002, sigma_max=80.0, 
                 sigma_data=0.302, rho=7.0, P_mean=-1.2, P_std=1.2, **kwargs):
        super().__init__(*args, **kwargs)
        # Noise schedule parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        
        # Noise sampling parameters
        self.P_mean = P_mean
        self.P_std = P_std
        
        # Sampling parameters
        self.S_churn = 80
        self.S_tmin = 0.05
        self.S_tmax = 50.
        self.S_noise = 1.003
    
    def get_scalings(self, sigma):
        """Get EDM network input/output scalings for given noise levels.
        
        Args:
            sigma: Noise levels
            
        Returns:
            c_skip: Skip connection scaling
            c_out: Output scaling
            c_in: Input scaling
            c_noise: Noise level embedding
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = sigma.log() * 0.25
        return c_skip, c_out, c_in, c_noise
    
    def noise(self, x_0, sigma):
        """Forward diffusion process using continuous sigma parameterization.
        
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
    
    def sample_noise_levels(self, batch_size):
        """Sample noise levels from log-normal distribution.
        
        Args:
            batch_size: Number of noise levels to sample
            
        Returns:
            sigma: Sampled noise levels
        """
        return torch.exp(
            torch.randn(batch_size, device=self.device) * self.P_std + self.P_mean
        )
    
    def loss_weight(self, sigma):
        """EDM loss weighting function.
        
        Args:
            sigma: Noise levels
            
        Returns:
            weights: Loss weights for each noise level
        """
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
    
    def compute_loss(self, x):
        """Training step using EDM formulation"""
        x = x.to(self.device)
        sigma = self.sample_noise_levels(x.shape[0])
        
        # Forward diffusion
        x_t, noise = self.noise(x, sigma)
        
        # Get scalings
        c_skip, c_out, c_in, c_noise = self.get_scalings(sigma)
        
        # Model prediction with conditioning
        model_output = self.model(
            x_t * c_in[:, None, None, None],
            c_noise
        )
        
        # Apply scalings
        pred = model_output * c_out[:, None, None, None]
        pred = pred + x_t * c_skip[:, None, None, None]
        
        # Weighted loss computation
        loss = F.mse_loss(pred, x, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])
        loss = (loss * self.loss_weight(sigma)).mean()
        
        return loss
    
    def sample_schedule(self):
        """Generate sampling schedule following Karras et al."""
        steps = torch.arange(self.nfe, device=self.device)
        sigmas = (
            self.sigma_max ** (1/self.rho) + 
            steps / (self.nfe-1) * 
            (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho))
        ) ** self.rho
        sigmas = F.pad(sigmas, (0, 1), value=0.)
        return sigmas
    
    @torch.no_grad()
    def _sample_impl(self, batch_size=16):
        """Sample new images using EDM sampling"""
        sigmas = self.sample_schedule()
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / self.nfe, sqrt(2) - 1),
            0.
        )
        x = torch.randn(batch_size, 1, 28, 28).to(self.device) * sigmas[0]
        
        for sigma, sigma_next, gamma in tqdm(zip(sigmas[:-1], sigmas[1:], gammas[:-1]), 
                                           desc='sampling', total=len(sigmas)-1):
            eps = self.S_noise * torch.randn_like(x) if gamma > 0 else 0
            sigma_hat = sigma + gamma * sigma
            x_hat = x + sqrt(sigma_hat**2 - sigma**2) * eps
            
            # Get model scalings
            c_skip, c_out, c_in, c_noise = self.get_scalings(
                torch.full((batch_size,), sigma_hat, device=self.device)
            )
            
            # Model prediction
            model_output = self.model(
                x_hat * c_in[:, None, None, None],
                c_noise
            )
            denoised = model_output * c_out[:, None, None, None]
            denoised = denoised + x_hat * c_skip[:, None, None, None]
            
            # Update x
            d = (x_hat - denoised) / sigma_hat
            dt = sigma_next - sigma_hat
            x = x_hat + d * dt
            
            # Second order correction
            if sigma_next > 0:
                # Get scalings for correction step
                c_skip, c_out, c_in, c_noise = self.get_scalings(
                    torch.full((batch_size,), sigma_next, device=self.device)
                )
                
                # Model prediction at next step
                model_output = self.model(
                    x * c_in[:, None, None, None],
                    c_noise
                )
                denoised_next = model_output * c_out[:, None, None, None]
                denoised_next = denoised_next + x * c_skip[:, None, None, None]
                
                # Apply correction
                d_next = (x - denoised_next) / sigma_next
                x = x_hat + 0.5 * dt * (d + d_next)
        
        return x.detach().cpu().clamp(0., 1.)