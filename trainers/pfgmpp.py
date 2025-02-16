from .base import BaseTrainer
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.beta import Beta
from tqdm import tqdm
from math import sqrt
from einops import rearrange

class PFGMppTrainer(BaseTrainer):
    """Probability Flow Generative Model++ trainer.
    
    Implementation of "PFGM++: Physics-Inspired Generative Models" 
    (Xu et al., 2023) https://arxiv.org/abs/2302.04265
    
    Combines concepts from diffusion models and electrostatics to learn
    a generative process through Poisson flow dynamics.
    """
    def __init__(self, *args, 
                 sigma_min=0.002, 
                 sigma_max=80.0,
                 sigma_data=0.302,
                 rho=7.0,
                 P_mean=-1.2,
                 P_std=1.2,
                 D=128,
                 S_churn=80,
                 S_tmin=0.05,
                 S_tmax=50.0,
                 S_noise=1.003,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # PFGM++ specific parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.D = D
        self.N = 1 * 28 * 28  # Input dimension
        
        # Sampling parameters
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    def log(self, t, eps=1e-20):
        return torch.log(t.clamp(min=eps))

    def beta_like(self, tensor, alpha, beta):
        """Generate Beta distributed samples matching input tensor shape"""
        dist = Beta(alpha, beta)
        return dist.sample(tensor.shape).to(tensor.device)

    def preconditioners(self, sigmas):
        """Compute network preconditioning parameters"""
        c_in = 1 * (sigmas ** 2 + self.sigma_data ** 2) ** -0.5
        c_noise = self.log(sigmas) * 0.25
        c_skip = (self.sigma_data ** 2) / (sigmas ** 2 + self.sigma_data ** 2)
        c_out = sigmas * self.sigma_data * (self.sigma_data ** 2 + sigmas ** 2) ** -0.5
        return c_in, c_noise, c_skip, c_out

    def noise(self, x: Tensor) -> Tensor:
        """PFGM++ forward process combining Beta and Gaussian noise"""
        # Sample log sigmas from normal distribution
        sigmas = torch.exp(torch.randn(x.shape[0], device=x.device) * self.P_std + self.P_mean)
        
        # Generate perturbation using PFGM++ process
        r = (sigmas * sqrt(self.D))[:, None, None, None]
        samples_norm = self.beta_like(x, self.N/2., self.D/2.).clip(1e-3, 1-1e-3)
        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        
        # Construct perturbation
        samples_norm = r * torch.sqrt(inverse_beta)
        samples_norm = rearrange(samples_norm, 'b c h w -> b (c h w)')
        gaussian = torch.randn_like(samples_norm)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        return x + rearrange(unit_gaussian * samples_norm, 'b (c h w) -> b c h w', c=1, h=28, w=28)

    def compute_loss(self, x: Tensor) -> Tensor:
        """Training objective with PFGM++ weighting"""
        x = x.to(self.device)
        sigmas = torch.exp(torch.randn(x.shape[0], device=x.device) * self.P_std + self.P_mean)
        
        # Apply forward process
        eps = self.noise(x) - x
        x_t = x + eps
        
        # Precondition model inputs
        c_in, c_noise, c_skip, c_out = self.preconditioners(sigmas)
        network_output = self.model(
            c_in[:, None, None, None] * x_t,
            c_noise
        )
        
        # Compute weighted loss
        x_pred = c_skip[:, None, None, None] * x_t + c_out[:, None, None, None] * network_output
        loss_weight = (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2
        return (F.mse_loss(x_pred, x, reduction='none').mean(dim=[1,2,3]) * loss_weight).mean()

    def sample_schedule(self):
        """Generate sampling schedule following PFGM++ paper"""
        steps = torch.arange(self.nfe, device=self.device)
        sigmas = (
            self.sigma_max ** (1/self.rho) + 
            steps / (self.nfe-1) * 
            (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho))
        ) ** self.rho
        return F.pad(sigmas, (0, 1), value=0.)

    @torch.no_grad()
    def _sample_impl(self, batch_size=16) -> Tensor:
        """Sampling with second-order PFGM++ ODE solver"""
        sigmas = self.sample_schedule()
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / self.nfe, sqrt(2) - 1),
            0.
        )
        
        # Initial noise
        x = torch.randn(batch_size, 1, 28, 28, device=self.device) * sigmas[0]
        
        # Sampling loop
        for sigma_t, sigma_next, gamma in tqdm(
            zip(sigmas[:-1], sigmas[1:], gammas[:-1]),
            desc='sampling', total=len(sigmas)-1
        ):
            # Stochastic sampling step
            eps = self.S_noise * torch.randn_like(x) if gamma > 0 else 0
            sigma_hat = sigma_t + gamma * sigma_t
            x_hat = x + sqrt(sigma_hat**2 - sigma_t**2) * eps
            
            # Model prediction
            c_in, c_noise, c_skip, c_out = self.preconditioners(
                torch.full((batch_size,), sigma_hat, device=self.device)
            )
            denoised = self.model(
                c_in[:, None, None, None] * x_hat,
                c_noise
            )
            denoised = c_skip[:, None, None, None] * x_hat + c_out[:, None, None, None] * denoised
            
            # ODE update
            d = (x_hat - denoised) / sigma_hat
            x = x_hat + d * (sigma_next - sigma_hat)
            
            # Second-order correction
            if sigma_next > 0:
                c_in, c_noise, c_skip, c_out = self.preconditioners(
                    torch.full((batch_size,), sigma_next, device=self.device)
                )
                denoised_next = self.model(
                    c_in[:, None, None, None] * x,
                    c_noise
                )
                denoised_next = c_skip[:, None, None, None] * x + c_out[:, None, None, None] * denoised_next
                d_next = (x - denoised_next) / sigma_next
                x = x_hat + 0.5 * (sigma_next - sigma_hat) * (d + d_next)
        
        return x.detach().cpu().clamp(0., 1.) 