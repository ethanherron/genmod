import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
import wandb
from networks.dit import DiT
from networks.flux_vae import VariationalEncoder
from trainers.ddpm import DDPMTrainer
from trainers.edm import EDMTrainer
from trainers.rectified_flow import RFTrainer
from trainers.variational_rf import VRFTrainer
from trainers.pfgmpp import PFGMppTrainer

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generative Model Training")
parser.add_argument('--model', type=str, required=True, 
                   choices=['ddpm', 'edm', 'rf', 'vrf', 'pfgmpp'],
                   help='Which model to train')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--num_steps', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)

# Model specific args
parser.add_argument('--num_timesteps', type=int, default=1000,
                   help='Number of diffusion steps (for diffusion models)')
parser.add_argument('--beta', type=float, default=1.0,
                   help='KL weight (for VRF)')
parser.add_argument('--sigma_min', type=float, default=0.002,
                   help='Min noise (for EDM/PFGM++)')
parser.add_argument('--sigma_max', type=float, default=80.0,
                   help='Max noise (for EDM/PFGM++)')
args = parser.parse_args()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation
tf = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST("/home/idealaboptimus/data/edherron", train=True, download=False, transform=tf)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    pin_memory=True,
)

# Model instantiation
def create_model(model_type):
    base_model = DiT(
        input_size=(28, 28),
        patch_size=4,
        in_channels=1,
        out_channels=1,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
    ).to(device)
    
    if model_type == 'vrf':
        encoder = VariationalEncoder(
            resolution=28,
            in_channels=3,
            ch=64,
            ch_mult=[1],
            num_res_blocks=2,
            z_channels=2,
        ).to(device)
        return base_model, encoder
    
    return base_model

def create_trainer(model_type, model, optimizer, device, dataloader):
    if model_type == 'ddpm':
        return DDPMTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            num_timesteps=args.num_timesteps
        )
    elif model_type == 'edm':
        return EDMTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            num_timesteps=args.num_timesteps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max
        )
    elif model_type == 'rf':
        return RFTrainer(
            velocity_model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader
        )
    elif model_type == 'vrf':
        model, encoder = model
        return VRFTrainer(
            velocity_model=model,
            encoder=encoder,
            beta_value=args.beta,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader
        )
    elif model_type == 'pfgmpp':
        return PFGMppTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            num_timesteps=args.num_timesteps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Create model and optimizer
    model = create_model(args.model)
    if args.model == 'vrf':
        model, encoder = model
        optimizer = optim.Adam(
            list(model.parameters()) + list(encoder.parameters()),
            lr=args.lr
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize wandb
    wandb.init(
        project="mnist",
        entity="genmod",
        name=args.model.upper(),
        config=vars(args)
    )
    
    # Create and run trainer
    trainer = create_trainer(
        args.model,
        model,
        optimizer,
        device,
        train_loader
    )
    trainer.train(args.num_steps)