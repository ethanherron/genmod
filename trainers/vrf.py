from .base import FlowTrainer
import torch
import torch.nn.functional as F

class VRFTrainer(FlowTrainer):
    """Variational Rectified Flow trainer.
    
    Extends Rectified Flow with a learned latent space through a 
    variational encoder, allowing for more expressive transformations
    while maintaining invertibility.
    """
    def __init__(self, velocity_model, encoder, beta_value, *args, **kwargs):
        """Initialize VRF trainer.
        
        Args:
            velocity_model: Model predicting velocity field
            encoder: Variational encoder model
            beta_value: Weight for KL divergence term
            *args, **kwargs: Additional arguments for FlowTrainer
        """
        super().__init__(velocity_model, *args, **kwargs)
        self.encoder = encoder  # Variational encoder
        self.beta = beta_value  # KL weight
    
    def compute_loss(self, x0, x1, xt, t, target):
        """Compute combined velocity matching and KL loss.
        
        Args:
            x0: Initial noise samples
            x1: Target data samples
            xt: Interpolated points at time t
            t: Timesteps in [0,1]
            target: Target velocities (x1 - x0)
            
        Returns:
            loss: Combined velocity matching and KL loss
        """
        # Encode trajectory information into latent space
        z = self.encoder(x0, x1, xt, t)
        
        # Predict velocity using both position and latent
        velocity = self.model(torch.cat([xt, z], dim=1), t)
        
        # Velocity matching loss
        velocity_field_loss = F.mse_loss(velocity, target)
        
        # KL regularization from encoder
        kl_loss = self.encoder.reg.kl
        
        # Combined loss with beta weighting
        return velocity_field_loss + (kl_loss * self.beta)