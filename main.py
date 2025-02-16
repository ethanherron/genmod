import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
import wandb
from networks.dit import DiT
from networks.flux_encoder import VariationalEncoder
from trainers.ddpm import DDPMTrainer
from trainers.vdm import VDMTrainer
from trainers.edm import EDMTrainer
from trainers.rf import RFTrainer
from trainers.vrf import VRFTrainer
from trainers.pfgmpp import PFGMppTrainer
from trainers.cm import CMTrainer
from trainers.cd import CDTrainer

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generative Model Training")
parser.add_argument('--model', type=str, required=True, 
                   choices=['ddpm', 'edm', 'rf', 'vrf', 'pfgmpp', 'cm', 'cd', 'vdm'],
                   help='Which model to train')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--num_steps', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--nfe', type=int, default=500,
                   help='Number of diffusion steps (for diffusion models)')
parser.add_argument('--beta', type=float, default=1.0,
                   help='KL weight (for VRF)')
parser.add_argument('--sigma_min', type=float, default=0.002,
                   help='Min noise (for EDM/PFGM++)')
parser.add_argument('--sigma_max', type=float, default=80.0,
                   help='Max noise (for EDM/PFGM++)')
parser.add_argument('--save_dir', type=str, default='results',
                   help='Directory to save generated samples')
parser.add_argument('--image_format', type=str, default='png',
                   choices=['png', 'jpg'], help='Output format for samples')
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
        in_channels=2 if model_type == 'vrf' else 1,
        out_channels=2 if model_type == 'vdm' else 1,
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
            nfe=args.nfe,
            save_dir=args.save_dir,
            image_format=args.image_format
        )
    elif model_type == 'edm':
        return EDMTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            nfe=args.nfe,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            save_dir=args.save_dir,
            image_format=args.image_format
        )
    elif model_type == 'vdm':
        return VDMTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            nfe=args.nfe,
            save_dir=args.save_dir,
            image_format=args.image_format
        )
    elif model_type == 'pfgmpp':
        return PFGMppTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            nfe=args.nfe,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            save_dir=args.save_dir,
            image_format=args.image_format
        )
    elif model_type == 'cm':
        return CMTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            nfe=args.nfe,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            save_dir=args.save_dir,
            image_format=args.image_format
        )
    elif model_type == 'cd':
        return CDTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            nfe=args.nfe,
            save_dir=args.save_dir,
            image_format=args.image_format
        )
    elif model_type == 'rf':
        return RFTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            nfe=args.nfe,
            save_dir=args.save_dir,
            image_format=args.image_format
        )
    elif model_type == 'vrf':
        model, encoder = model
        return VRFTrainer(
            model=model,
            encoder=encoder,
            beta_value=args.beta,
            optimizer=optimizer,
            device=device,
            dataloader=dataloader,
            nfe=args.nfe,
            save_dir=args.save_dir,
            image_format=args.image_format
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Create model and optimizer
    if args.model == 'vrf':
        model, encoder = create_model(args.model)
        optimizer = optim.Adam(
            list(model.parameters()) + list(encoder.parameters()),
            lr=args.lr
        )
    else:
        model = create_model(args.model)
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
        model if not args.model == 'vrf' else (model, encoder),
        optimizer,
        device,
        train_loader
    )
    trainer.train(args.num_steps)
    trainer.sample()