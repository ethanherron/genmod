from .base import BaseTrainer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from math import log

class VDMTrainer(BaseTrainer):
    """Variational Diffusion Model trainer.
    
    Implementation of "Variational Diffusion Models" (Kingma et al., 2021)
    https://arxiv.org/abs/2107.00630
    
    Uses continuous SNR parameterization and optimizes the variational lower bound
    directly. The model predicts both mean and variance of the reverse process,
    allowing for more flexible noise schedules.
    """
    def __init__(self, *args, min_snr=0.001, max_snr=1000, **kwargs):
        super().__init__(*args, **kwargs)
        # SNR schedule parameters
        self.min_snr = min_snr  # Minimum signal-to-noise ratio
        self.max_snr = max_snr  # Maximum signal-to-noise ratio
        self.register_schedules()
        
    def register_schedules(self):
        """Pre-compute VDM noise schedules using SNR parameterization.
        
        Creates a continuous schedule of signal-to-noise ratios that
        interpolates between max_snr and min_snr on a logarithmic scale.
        """
        # Time grid from log-SNR schedule
        log_snr = torch.linspace(
            log(self.max_snr), 
            log(self.min_snr), 
            self.nfe
        ).to(self.device)
        
        # Convert to alpha values for compatibility
        alphas = torch.sigmoid(log_snr)
        alphas_cumprod = alphas.cumprod(dim=0)
        
        # Store values
        self.register_buffer('log_snr', log_snr)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def register_buffer(self, name, tensor):
        """Helper to register buffer on device"""
        setattr(self, name, tensor)
    
    def get_snr(self, t):
        """Get SNR value for timestep t.
        
        Args:
            t: Timestep indices
            
        Returns:
            snr: Signal-to-noise ratios at timesteps t
        """
        return self.log_snr[t].exp()
    
    def noise(self, x_0, t):
        """Forward diffusion process using SNR parameterization.
        
        Args:
            x_0: Clean images
            t: Timesteps
            
        Returns:
            x_t: Noisy images
            noise: Added noise (for training)
        """
        noise = torch.randn_like(x_0)
        snr = self.get_snr(t)[:, None, None, None]
        
        # Mean and variance for q(x_t | x_0)
        mean = x_0 / torch.sqrt(1 + 1/snr)
        var = 1 / (1 + snr)
        std = torch.sqrt(var)
        
        x_t = mean + std * noise
        return x_t, noise
    
    def kl_prior(self, mean, log_var):
        """KL divergence with standard normal prior.
        
        Args:
            mean: Predicted mean
            log_var: Predicted log variance
            
        Returns:
            kl: KL divergence term
        """
        return -0.5 * torch.sum(
            1 + log_var - mean.pow(2) - log_var.exp(),
            dim=[1, 2, 3]
        ).mean()
    
    def compute_loss(self, x):
        """Training step optimizing variational lower bound"""
        x = x.to(self.device)
        # Sample random timesteps
        t = torch.randint(0, self.nfe, (x.shape[0],), device=self.device)
        
        # Forward diffusion
        x_t, noise = self.noise(x, t)
        
        # Model predicts mean and log variance
        pred_mean, pred_log_var = self.model(x_t, t / self.nfe).chunk(2, dim=1)
        
        # Get target mean using noise prediction
        snr = self.get_snr(t)[:, None, None, None]
        target_mean = -noise / torch.sqrt(1 + snr)
        
        # Compute losses
        mean_loss = F.mse_loss(pred_mean, target_mean)
        var_loss = self.kl_prior(pred_mean, pred_log_var)
        
        # Total loss is sum of mean and variance terms
        loss = mean_loss + 0.001 * var_loss
        return loss
    
    @torch.no_grad()
    def _sample_impl(self, batch_size=16):
        """Sample new images using learned variance.
        
        Uses the predicted mean and variance for each step,
        allowing for adaptive step sizes during sampling.
        """
        # Start from pure noise
        x = torch.randn(batch_size, 1, 28, 28).to(self.device)
        
        # Reverse process
        for t in tqdm(reversed(range(self.nfe)), desc='sampling', total=self.nfe):
            t_batch = torch.full((batch_size,), t, device=self.device)
            
            # Get model predictions
            pred_mean, pred_log_var = self.model(
                x, t_batch / self.nfe
            ).chunk(2, dim=1)
            
            # Sample from predicted distribution
            if t > 0:
                noise = torch.randn_like(x)
                x = pred_mean + torch.exp(0.5 * pred_log_var) * noise
            else:
                x = pred_mean
        
        return x.detach().cpu().clamp(0., 1.) 