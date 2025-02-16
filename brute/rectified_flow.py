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
parser = argparse.ArgumentParser(description="Rectified Flow Training Script")
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
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

# Trainer class definition
class Trainer:
    def __init__(self, velocity_model, optimizer, device, dataloader):
        self.velocity_model = velocity_model
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
        
    def train_step(self, x1: Tensor):
        x1 = x1.to(self.device)
        t = torch.randn(x1.shape[0], device=self.device).sigmoid()
        x0 = torch.randn_like(x1)
        xt = torch.lerp(x0, x1, t[:, None, None, None])
        target = x1 - x0
        velocity = self.velocity_model(xt, t)
        loss = F.mse_loss(velocity, target)
        return loss
    
    def train(self, num_steps):
        self.velocity_model.train()
        loader = iter(self.dataloader)
        for step in range(num_steps):
            try:
                x1, _ = next(loader)
            except StopIteration:
                loader = iter(self.dataloader)
                x1, _ = next(loader)
            # Unpack tuple from train_step
            loss = self.train_step(x1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Log metrics to wandb
            wandb.log({
                "total_loss": loss.item(),
                "step": step,
            })
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss.item():.6f}")

# Model instantiation
velocity_model = DiT(
    input_size=(28, 28),
    patch_size=4,
    in_channels=1,
    out_channels=1,
    hidden_size=384,
    depth=6,
    num_heads=6,
    mlp_ratio=4.0,
).to(device)

# Create an optimizer for both the velocity model and the encoder
optimizer = optim.Adam(
    list(velocity_model.parameters()),
    lr=args.lr,
)

if __name__ == "__main__":
    # Initialize wandb with the specified project, entity, and run name
    wandb.init(project="mnist", entity="genmod", name="RF")
    trainer = Trainer(
        velocity_model=velocity_model,
        optimizer=optimizer,
        device=device,
        dataloader=train_loader,
    )
    trainer.train(args.num_steps)