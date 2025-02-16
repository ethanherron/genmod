from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch import Tensor
import wandb
from tqdm import tqdm
from torchvision.utils import save_image
import os

class BaseTrainer:
    """Base trainer for all generative models.
    
    This class provides the basic training loop and utilities common to all models.
    Specific model implementations should inherit from this or its subclasses.
    """
    def __init__(self, model, optimizer, device, dataloader, nfe: int = 50, 
                 save_dir: str = None, image_format: str = 'png'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
        self.nfe = nfe
        self.save_dir = save_dir
        self.image_format = image_format
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
    @abstractmethod
    def noise(self, x: Tensor) -> Tensor:
        """Method to find noisy sample xt from x0 and t.
        In diffusion models, this is the forward process.
        In flow models, this is the interpolation process.
        
        Args:
            x: Input batch of images
            
        Returns:
            xt: Noisy images
        """
        raise NotImplementedError
        
    @abstractmethod
    def compute_loss(self, x: Tensor) -> Tensor:
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
            loss = self.compute_loss(x)
            
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
            
    def sample(self, batch_size: int = 16, save: bool = True) -> Tensor:
        """Sampling process - implement in subclass.
        
        Args:
            batch_size: Number of images to sample
            save: Whether to save samples to disk
            
        Returns:
            samples: Generated images
        """
        samples = self._sample_impl(batch_size)
        
        if save and self.save_dir is not None:
            method_name = self.__class__.__name__.lower().replace("trainer", "")
            filename = f"{method_name}_generated_samples.{self.image_format}"
            save_path = os.path.join(self.save_dir, filename)
            save_image(samples, save_path, normalize=True)
            
        return samples
    
    @abstractmethod
    def _sample_impl(self, batch_size: int) -> Tensor:
        """Implementation-specific sampling logic"""
        raise NotImplementedError
    
    