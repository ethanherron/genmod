import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST
import wandb
from networks.dit import DiT

# Parse command line arguments
parser = argparse.ArgumentParser(description="Denoising Diffusion Probabilistic Model Training Script")
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion steps')
parser.add_argument('--num_steps', type=int, default=1000, help='Number of training steps')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
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

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class Trainer:
    def __init__(self, model, optimizer, device, dataloader, num_timesteps):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
        self.n_T = num_timesteps
        self.register_schedules()
        
    def register_schedules(self):
        """Pre-compute diffusion schedules"""
        beta_t = linear_beta_schedule(self.n_T)
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)
        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("oneover_sqrta", oneover_sqrta)
        self.register_buffer("sqrt_beta_t", sqrt_beta_t)
        self.register_buffer("alphabar_t", alphabar_t)
        self.register_buffer("sqrtab", sqrtab)
        self.register_buffer("sqrtmab", sqrtmab)
        self.register_buffer("mab_over_sqrtmab", mab_over_sqrtmab_inv)
        
    def register_buffer(self, name, tensor):
        """Helper to register buffer on device"""
        setattr(self, name, tensor.to(self.device))

    def forward_diffusion(self, x, t, noise):
        x_t = (self.sqrtab[t, None, None, None] * x + 
               self.sqrtmab[t, None, None, None] * noise)
        return x_t
    
    def train_step(self, x: Tensor):
        x = x.to(self.device)
        noise = torch.randn_like(x)
        t = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)
        x_t = self.forward_diffusion(x, t, noise)
        predicted_noise = self.model(x_t, t / self.n_T)
        loss = F.mse_loss(noise, predicted_noise)
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
    
    def sample(self, batch_size=16):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(batch_size, 1, 28, 28).to(self.device)
            
            for i in range(self.n_T-1, 0, -1):
                t = torch.ones(batch_size).to(self.device) * i / self.n_T
                predicted_noise = self.model(x, t)
                
                # No noise for last step
                z = torch.randn_like(x) if i > 1 else 0
                
                x = (self.oneover_sqrta[i] * 
                     (x - predicted_noise * self.mab_over_sqrtmab[i]) +
                     self.sqrt_beta_t[i] * z)
                
            return x

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
    wandb.init(project="mnist", entity="genmod", name="DDPM")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        dataloader=train_loader,
        num_timesteps=args.num_timesteps
    )
    trainer.train(args.num_steps)
