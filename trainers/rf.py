from .base import FlowTrainer
import torch.nn.functional as F

class RFTrainer(FlowTrainer):
    """Rectified Flow trainer.
    
    Implementation of "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" 
    (Liu et al., 2023) https://arxiv.org/abs/2209.03003
    
    Learns a velocity field that transports noise to data through
    a deterministic continuous-time flow, using linear interpolation
    between distributions.
    """
    def compute_loss(self, x0, x1, xt, t, target):
        """Compute velocity matching loss.
        
        Args:
            x0: Initial noise samples
            x1: Target data samples
            xt: Interpolated points at time t
            t: Timesteps in [0,1]
            target: Target velocities (x1 - x0)
            
        Returns:
            loss: MSE between predicted and target velocities
        """
        # Predict velocity field at interpolated points
        velocity = self.model(xt, t)
        
        # Match predicted velocity to target direction
        return F.mse_loss(velocity, target)