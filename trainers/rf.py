from .base import BaseTrainer
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

class RFTrainer(BaseTrainer):
    """Rectified Flow trainer.
    
    Implementation of "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" 
    (Liu et al., 2023) https://arxiv.org/abs/2209.03003
    
    Learns a velocity field that transports noise to data through
    a deterministic continuous-time flow, using linear interpolation
    between distributions.
    """
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
        return xt, t, target
    
    def compute_loss(self, x: Tensor) -> Tensor:
        """Compute velocity matching loss.
        
        Args:
            x: Target data samples
            
        Returns:
            loss: MSE between predicted and target velocities
        """
        xt, t, target = self.noise(x)
        velocity = self.model(xt, t)
        return F.mse_loss(velocity, target)
    
    @torch.no_grad()
    def _sample_impl(self, batch_size: int = 16) -> Tensor:
        """Sampling process for Rectified Flow.
        Use Euler method to integrate the flow field from x0 ~ N(0, I) to x1 ~ p_data.
        
        Args:
            batch_size: Number of images to sample
            
        Returns:
            samples: Generated images
        """
        dt = 1 / self.nfe
        x = torch.randn((batch_size, 1, 28, 28), device=self.device)
        x = x.detach().clone()
        for _ in tqdm(range(self.nfe), desc='sampling', total=self.nfe):
            t = torch.ones((batch_size,), device=self.device) * (1 / self.nfe)
            velocity = self.model(x, t)
            x = x.detach().clone() + velocity * dt
        return x.detach().cpu().clip(0., 1.)
        