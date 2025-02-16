import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.distributions.beta import Beta
from torchvision import transforms
from torchvision.datasets import MNIST
import wandb
from networks.dit import DiT
from tqdm import tqdm
from math import sqrt
from einops import rearrange

# Parse command line arguments
parser = argparse.ArgumentParser(description="Probability Flow Generative Model++ Training Script")
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
parser.add_argument('--num_timesteps', type=int, default=50, help='Number of diffusion steps')
parser.add_argument('--num_steps', type=int, default=1000, help='Number of training steps')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--sigma_min', type=float, default=0.002, help='Min noise level')
parser.add_argument('--sigma_max', type=float, default=80.0, help='Max noise level')
parser.add_argument('--sigma_data', type=float, default=0.302, help='Data distribution std')
parser.add_argument('--rho', type=float, default=7.0, help='Controls sampling schedule')
parser.add_argument('--P_mean', type=float, default=-1.2, help='Mean of noise distribution')
parser.add_argument('--P_std', type=float, default=1.2, help='Std of noise distribution')
parser.add_argument('--D', type=int, default=128, help='Dimension parameter')
args = parser.parse_args()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation
tf = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST("/home/idealaboptimus/data/edherron", train=True, download=False, transform=tf)
val_dataset = MNIST("/home/idealaboptimus/data/edherron", train=False, download=False, transform=tf)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def beta_like(tensor, alpha, beta):
    """Generate Beta distributed samples with same shape as input tensor"""
    shape = tensor.shape
    dist = Beta(alpha, beta)
    return dist.sample(shape)

class Trainer:
    def __init__(self, model, optimizer, device, dataloader, num_timesteps, 
                 sigma_min, sigma_max, sigma_data, rho, P_mean, P_std, D):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
        self.n_T = num_timesteps
        
        # PFGM++ specific parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.D = D
        self.N = 1 * 28 * 28  # MNIST dimension
        
        # Sampling parameters
        self.S_churn = 80
        self.S_tmin = 0.05
        self.S_tmax = 50.
        self.S_noise = 1.003
        
    def preconditioners(self, sigmas):
        """Compute PFGM++ preconditioners"""
        c_in = 1 * (sigmas ** 2 + self.sigma_data ** 2) ** -0.5
        c_noise = log(sigmas) * 0.25
        c_skip = (self.sigma_data ** 2) / (sigmas ** 2 + self.sigma_data ** 2)
        c_out = sigmas * self.sigma_data * (self.sigma_data ** 2 + sigmas ** 2) ** -0.5
        return c_in, c_noise, c_skip, c_out
    
    def noise_distribution(self, x):
        """Sample noise levels for training"""
        return (self.P_mean + self.P_std * torch.randn((x.shape[0],))).exp()
    
    def forward_diffusion(self, x_0, sigmas):
        """Generate perturbation using PFGM++ noise process"""
        # Sample radii
        r = (sigmas * sqrt(self.D)).to(x_0)
        
        # Sample inverse-beta distribution
        samples_norm = beta_like(x_0, self.N / 2., self.D / 2.).to(x_0).clip(1e-3, 1-1e-3)
        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        
        # Sample from p_r(R) by change-of-variables
        samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
        samples_norm = rearrange(samples_norm, 'b c h w -> b (c h w)')
        
        # Sample angle direction uniformly
        gaussian = torch.randn(x_0.shape[0], self.N).to(x_0)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        
        # Construct perturbation
        perturbation = (unit_gaussian * samples_norm)
        eps = rearrange(perturbation, 'b (c h w) -> b c h w', c=1, h=28, w=28)
        return eps
    
    def loss_weight(self, sigmas):
        """Compute loss weights"""
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2
    
    def train_step(self, x: Tensor):
        x = x.to(self.device)
        sigmas = self.noise_distribution(x)[:, None, None, None].to(self.device)
        
        # Generate perturbation and add to input
        eps = self.forward_diffusion(x, sigmas)
        x_t = x + eps
        
        # Apply model with preconditioners
        c_in, c_noise, c_skip, c_out = self.preconditioners(sigmas.squeeze())
        network_output = self.model(
            (c_in[:, None, None, None] * x_t),
            c_noise
        )
        x_pred = c_skip[:, None, None, None] * x_t + c_out[:, None, None, None] * network_output
        
        # Compute loss
        loss = F.mse_loss(x_pred, x, reduction='none')
        loss = (self.loss_weight(sigmas)[:, None, None, None] * loss).mean()
        return loss
    
    def train(self, num_steps):
        self.model.train()
        loader = iter(self.dataloader)
        for step in range(num_steps):
            try:
                x, _ = next(loader)
            except StopIteration:
                loader = iter(self.dataloader)
                x, _ = next(loader)
                
            loss = self.train_step(x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            wandb.log({
                "loss": loss.item(),
                "step": step,
            })
            
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss.item():.6f}")
    
    def sample_schedule(self):
        """Generate sampling schedule"""
        steps = torch.arange(self.n_T, device=self.device)
        sigmas = (
            self.sigma_max ** (1/self.rho) + 
            steps / (self.n_T-1) * 
            (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho))
        ) ** self.rho
        sigmas = F.pad(sigmas, (0, 1), value=0.)
        return sigmas
    
    def sample(self, batch_size=16):
        self.model.eval()
        with torch.no_grad():
            # Initialize with noise
            sigmas = self.sample_schedule()
            gammas = torch.where(
                (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
                min(self.S_churn / self.n_T, sqrt(2) - 1),
                0.
            )
            
            # Initial noise
            x = self.forward_diffusion(
                torch.zeros(batch_size, 1, 28, 28).to(self.device),
                sigmas[0]
            )
            
            # Sampling loop
            for sigma_t, sigma_next, gamma in tqdm(
                zip(sigmas[:-1], sigmas[1:], gammas[:-1]), 
                desc='sampling'
            ):
                # Add noise if specified by gamma
                eps = self.S_noise * torch.randn_like(x) if gamma > 0 else 0
                sigma_hat = sigma_t + gamma * sigma_t
                x_hat = x + sqrt(sigma_hat**2 - sigma_t**2) * eps
                
                # Predict and update
                c_in, c_noise, c_skip, c_out = self.preconditioners(sigma_hat.repeat(batch_size))
                denoised = self.model(
                    (c_in[:, None, None, None] * x_hat),
                    c_noise[:, None, None, None]
                )
                denoised = c_skip[:, None, None, None] * x_hat + c_out[:, None, None, None] * denoised
                d = (x_hat - denoised) / sigma_hat
                
                # Update x
                dt = sigma_next - sigma_hat
                x = x_hat + d * dt
                
                # Second order correction
                if sigma_next > 0:
                    c_in, c_noise, c_skip, c_out = self.preconditioners(sigma_next.repeat(batch_size))
                    denoised_next = self.model(
                        (c_in[:, None, None, None] * x),
                        c_noise[:, None, None, None]
                    )
                    denoised_next = c_skip[:, None, None, None] * x + c_out[:, None, None, None] * denoised_next
                    d_next = (x - denoised_next) / sigma_next
                    x = x_hat + 0.5 * dt * (d + d_next)
            
            return x.clamp(0., 1.)

# Model instantiation
model = DiT(
    input_size=(28, 28),
    patch_size=4,
    in_channels=1,
    out_channels=1,
    hidden_size=384,
    depth=6,
    num_heads=6,
    mlp_ratio=4.0,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

if __name__ == "__main__":
    wandb.init(project="mnist", entity="genmod", name="PFGM++")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        dataloader=train_loader,
        num_timesteps=args.num_timesteps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
        rho=args.rho,
        P_mean=args.P_mean,
        P_std=args.P_std,
        D=args.D
    )
    trainer.train(args.num_steps)
