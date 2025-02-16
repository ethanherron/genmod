from .base import DiffusionTrainer
import torch
import torch.nn.functional as F
from tqdm import tqdm

class CDTrainer(DiffusionTrainer):
    """Cold Diffusion trainer.
    
    Implementation of "Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise"
    (Bansal et al., 2022) https://arxiv.org/abs/2208.09392
    
    Replaces stochastic noise-based diffusion with deterministic image degradation.
    This implementation uses Gaussian blur as the degradation operator, but the method
    can work with any invertible image transformation.
    """
    def __init__(self, *args, kernel_size=11, kernel_std=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        # Blur parameters
        self.kernel_size = kernel_size
        self.kernel_std = kernel_std
        self.register_kernels()
        
    def register_kernels(self):
        """Pre-compute Gaussian blur kernels for each timestep.
        
        Creates a sequence of increasingly strong blur kernels,
        one for each timestep in the diffusion process.
        """
        kernels = []
        for t in range(self.n_T):
            # Create Gaussian kernel with increasing std over time
            std = self.kernel_std * (t + 1)
            kernel = self._get_gaussian_kernel2d(
                (self.kernel_size, self.kernel_size), 
                (std, std)
            )
            # Expand kernel for group convolution
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernels.append(kernel)
            
        # Stack all kernels
        self.register_buffer('kernels', torch.stack(kernels))
    
    def _get_gaussian_kernel2d(self, kernel_size, sigma):
        """Create 2D Gaussian kernel.
        
        Args:
            kernel_size: Tuple of kernel height and width
            sigma: Tuple of standard deviations (σx, σy)
            
        Returns:
            kernel: 2D Gaussian kernel
        """
        kx = torch.arange(
            -(kernel_size[0] - 1) // 2, 
            (kernel_size[0] - 1) // 2 + 1, 
            dtype=torch.float32
        )
        ky = torch.arange(
            -(kernel_size[1] - 1) // 2, 
            (kernel_size[1] - 1) // 2 + 1, 
            dtype=torch.float32
        )
        
        # Create 2D Gaussian distribution
        x, y = torch.meshgrid(kx, ky)
        kernel = torch.exp(-(x.pow(2)/(2*sigma[0]**2) + y.pow(2)/(2*sigma[1]**2)))
        return kernel / kernel.sum()
    
    def apply_blur(self, x, t):
        """Apply Gaussian blur at timestep t.
        
        Args:
            x: Input images
            t: Timestep index
            
        Returns:
            blurred: Blurred images
        """
        # Pad input for circular convolution
        pad_size = self.kernel_size // 2
        x_pad = F.pad(x, (pad_size,)*4, mode='circular')
        
        # Apply convolution for each channel
        blurred = []
        for i in range(x.size(1)):
            channel = x_pad[:, i:i+1]
            kernel = self.kernels[t].to(x.device)
            blurred.append(F.conv2d(channel, kernel, padding=0))
        return torch.cat(blurred, dim=1)
    
    def forward_diffusion(self, x_0, t):
        """Forward process: apply increasing blur.
        
        Args:
            x_0: Clean images
            t: Timesteps
            
        Returns:
            x_t: Blurred images at timesteps t
        """
        x_t = x_0.clone()
        for i in range(t.max().item() + 1):
            mask = t >= i
            if not mask.any():
                break
            x_t[mask] = self.apply_blur(x_t[mask], i)
        return x_t
    
    def train_step(self, x):
        """Training step predicting clean images from blurred ones"""
        x = x.to(self.device)
        # Sample random timesteps
        t = torch.randint(0, self.n_T, (x.shape[0],), device=self.device)
        
        # Apply forward process (blur)
        x_t = self.forward_diffusion(x, t)
        
        # Model predicts clean image
        pred = self.model(x_t, t / self.n_T)
        
        # Loss is MSE between prediction and original
        return F.mse_loss(pred, x)
    
    def sample(self, batch_size=16):
        """Sample new images using Cold Diffusion.
        
        Implements Algorithm 2 from the paper, using the difference
        between consecutive blur levels to guide the reverse process.
        """
        self.model.eval()
        with torch.no_grad():
            # Start from random noise
            x = torch.randn(batch_size, 1, 28, 28).to(self.device)
            
            # Apply maximum blur
            x = self.forward_diffusion(
                x, 
                torch.full((batch_size,), self.n_T-1, device=self.device)
            )
            
            # Reverse process
            for t in tqdm(reversed(range(self.n_T)), desc='sampling'):
                t_batch = torch.full((batch_size,), t, device=self.device)
                
                # Get model prediction of clean image
                x_0_hat = self.model(x, t_batch / self.n_T)
                
                if t > 0:
                    # Get blurred versions of prediction
                    x_t = self.forward_diffusion(x_0_hat, t_batch)
                    x_t_minus_1 = self.forward_diffusion(x_0_hat, t_batch - 1)
                    
                    # Update using difference between blur levels
                    x = x - x_t + x_t_minus_1
                else:
                    x = x_0_hat
            
            return x.clamp(0., 1.) 