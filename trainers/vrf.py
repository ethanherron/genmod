from .base import BaseTrainer
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

class VRFTrainer(BaseTrainer):
    """Variational Rectified Flow trainer.
    
    Implementation of "Variational Rectified Flow" (Guo et al., 2025)
    https://arxiv.org/pdf/2502.09616
    
    Extends Rectified Flow with a learned ~conditional-ish term, which assists model
    in learning a multi-modal velocity field.
    """
    def __init__(self, model, encoder, beta_value, *args, **kwargs):
        """Initialize VRF trainer.
        
        Args:
            model: Model predicting velocity field
            encoder: Variational encoder model
            beta_value: Weight for KL divergence term
            *args, **kwargs: Additional arguments for FlowTrainer
        """
        super().__init__(model, *args, **kwargs)
        self.encoder = encoder  # Variational encoder
        self.beta = beta_value  # KL weight
    
    def noise(self, x: Tensor) -> Tensor:
        """Noise function for Rectified Flow.
        
        Args:
            x: Input data samples
            
        Returns:
            xt: Noisy images
            t: Timesteps in [0,1]
            target: Target velocities (x1 - x0)
        """
        x1 = x.to(self.device)
        x0 = torch.randn_like(x1)
        t = torch.randn((x.shape[0],), device=self.device).sigmoid()
        xt = torch.lerp(x0, x1, t[:, None, None, None])
        target = x1 - x0
        z = self.encoder(x0, x1, xt, t)
        return xt, t, target, z
    
    def compute_loss(self, x: Tensor) -> Tensor:
        """Compute velocity matching loss.
        
        Args:
            x: Target data samples
            
        Returns:
            loss: MSE between predicted and target velocities
        """
        xt, t, target, z = self.noise(x)
        velocity = self.model(torch.cat([xt, z], dim=1), t)
        velocity_field_loss = F.mse_loss(velocity, target)
        kl_loss = self.encoder.reg.kl
        return velocity_field_loss + (kl_loss * self.beta)
    
    @torch.no_grad()
    def _sample_impl(self, batch_size: int = 16) -> Tensor:
        """Sampling process for Variational Rectified Flow.
        Use Euler method to integrate the flow field from x0 ~ N(0, I) to x1 ~ p_data.
        
        Args:
            batch_size: Number of images to sample
            
        Returns:
            samples: Generated images
        """
        dt = 1 / self.nfe
        x = torch.randn((batch_size, 1, 28, 28), device=self.device)
        x = x.detach().clone()
        z = torch.randn((batch_size, 1, 28, 28), device=self.device).detach()
        for _ in tqdm(range(self.nfe), desc='sampling', total=self.nfe):
            t = torch.ones((batch_size,), device=self.device) * (1 / self.nfe)
            velocity = self.model(torch.cat([x, z], dim=1), t)
            x = x.detach().clone() + velocity * dt
        return x.detach().cpu().clip(0., 1.)