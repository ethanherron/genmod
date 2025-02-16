from .base import BaseTrainer
import torch
import torch.nn.functional as F
from tqdm import tqdm

class DDPMTrainer(BaseTrainer):
    """Denoising Diffusion Probabilistic Models trainer.
    
    Implementation of "Denoising Diffusion Probabilistic Models" 
    (Ho et al., 2020) https://arxiv.org/abs/2006.11239
    
    The model learns to predict noise in images at different timesteps,
    allowing generation through gradual denoising.
    """
    def __init__(self, *args, beta_start=0.0001, beta_end=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_schedules(beta_start, beta_end)
        
    def register_schedules(self, beta_start, beta_end):
        """Pre-compute DDPM noise schedules"""
        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, self.nfe).to(self.device)
        
        # Pre-compute diffusion values
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Store values
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def register_buffer(self, name, tensor):
        """Helper to register buffer on device"""
        setattr(self, name, tensor)
    
    def noise(self, x_0, t):
        """Forward diffusion process q(x_t | x_0)"""
        noise = torch.randn_like(x_0)
        mean = (self.alphas_cumprod[t] ** 0.5)[:, None, None, None] * x_0
        var = (1 - self.alphas_cumprod[t])[:, None, None, None]
        return mean + (var ** 0.5) * noise, noise
    
    def compute_loss(self, x):
        """Training step predicting noise at random timesteps"""
        x = x.to(self.device)
        t = torch.randint(0, self.nfe, (x.shape[0],), device=self.device)
        
        # Forward diffusion
        x_t, noise = self.noise(x, t)
        
        # Predict and optimize
        pred_noise = self.model(x_t, t / self.nfe)
        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def _sample_impl(self, batch_size=16):
        """Sample new images"""
        # Start from pure noise
        x = torch.randn(batch_size, 1, 28, 28).to(self.device)
        
        # Reverse diffusion process
        for t in tqdm(reversed(range(self.nfe)), desc='sampling', total=self.nfe):
            t_tensor = torch.tensor([t], device=self.device)
            t_tensor = t_tensor.repeat(batch_size)
            
            # Predict noise
            predicted_noise = self.model(x, t_tensor / self.nfe)
            
            # Get alpha values for this timestep
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # No noise for last step
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # Reverse diffusion step
            x = (
                1 / torch.sqrt(alpha) * 
                (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) +
                torch.sqrt(beta) * noise
            )
        
        return x.detach().cpu().clamp(0., 1.) 