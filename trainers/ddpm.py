from .base import DiffusionTrainer
import torch
import torch.nn.functional as F

class DDPMTrainer(DiffusionTrainer):
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
        betas = torch.linspace(beta_start, beta_end, self.n_T).to(self.device)
        
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
    
    def forward_diffusion(self, x_0, t):
        """Forward diffusion process q(x_t | x_0)"""
        noise = torch.randn_like(x_0)
        mean = (self.alphas_cumprod[t] ** 0.5)[:, None, None, None] * x_0
        var = (1 - self.alphas_cumprod[t])[:, None, None, None]
        return mean + (var ** 0.5) * noise, noise
    
    def train_step(self, x):
        """Training step predicting noise at random timesteps"""
        x = x.to(self.device)
        t = torch.randint(0, self.n_T, (x.shape[0],), device=self.device)
        
        # Forward diffusion
        x_t, noise = self.forward_diffusion(x, t)
        
        # Predict and optimize
        pred_noise = self.model(x_t, t / self.n_T)
        return F.mse_loss(pred_noise, noise)
    
    def sample(self, batch_size=16):
        """Sample new images"""
        self.model.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(batch_size, 1, 28, 28).to(self.device)
            
            # Reverse diffusion process
            for t in reversed(range(self.n_T)):
                t_tensor = torch.tensor([t], device=self.device)
                t_tensor = t_tensor.repeat(batch_size)
                
                # Predict noise
                predicted_noise = self.model(x, t_tensor / self.n_T)
                
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
            
            return x.clamp(0., 1.) 