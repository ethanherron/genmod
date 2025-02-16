from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch import Tensor
import wandb
from tqdm import tqdm

class BaseTrainer:
    """Base trainer for all generative models.
    
    This class provides the basic training loop and utilities common to all models.
    Specific model implementations should inherit from this or its subclasses.
    """
    def __init__(self, model, optimizer, device, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
    
    @abstractmethod
    def train_step(self, x: Tensor) -> Tensor:
        """Single training step - implement in subclass.
        
        Args:
            x: Input batch of images
            
        Returns:
            loss: Scalar training loss
        """
        raise NotImplementedError
    
    def train(self, num_steps: int):
        """Common training loop for all models.
        
        Args:
            num_steps: Number of training steps to run
        """
        self.model.train()
        pbar = tqdm(range(num_steps), desc='training')
        loader = iter(self.dataloader)
        
        for step in pbar:
            # Get next batch, restarting if needed
            try:
                x, _ = next(loader)
            except StopIteration:
                loader = iter(self.dataloader)
                x, _ = next(loader)
            
            # Training step    
            loss = self.train_step(x)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Logging
            wandb.log({
                "loss": loss.item(),
                "step": step,
            })
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

class DiffusionTrainer(BaseTrainer):
    """Base class for diffusion-based models (DDPM, EDM, VDM).
    
    Implements common functionality for models that use a noise-based
    forward process and gradual denoising.
    """
    def __init__(self, model, optimizer, device, dataloader, num_timesteps):
        super().__init__(model, optimizer, device, dataloader)
        self.n_T = num_timesteps
    
    @abstractmethod
    def forward_diffusion(self, x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Forward diffusion process - implement in subclass.
        
        Args:
            x_0: Clean images
            t: Timesteps
            
        Returns:
            x_t: Noisy images
            noise: Added noise (for training)
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, batch_size: int = 16) -> Tensor:
        """Sampling process - implement in subclass.
        
        Args:
            batch_size: Number of images to sample
            
        Returns:
            samples: Generated images
        """
        raise NotImplementedError

class FlowTrainer(BaseTrainer):
    """Base class for flow-based models (RF, VRF).
    
    Implements common functionality for models that learn
    a velocity field between distributions.
    """
    def train_step(self, x1: Tensor) -> Tensor:
        """Common training step for flow models.
        
        Args:
            x1: Target images
            
        Returns:
            loss: Training loss
        """
        x1 = x1.to(self.device)
        t = torch.randn(x1.shape[0], device=self.device).sigmoid()
        x0 = torch.randn_like(x1)
        xt = torch.lerp(x0, x1, t[:, None, None, None])
        target = x1 - x0
        return self.compute_loss(x0, x1, xt, t, target)
    
    @abstractmethod
    def compute_loss(self, x0, x1, xt, t, target):
        """Compute loss - implement in subclass"""
        raise NotImplementedError