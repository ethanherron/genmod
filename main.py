import argparse, os
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms
from torchvision.datasets import MNIST

from modules.trainers.DDPM import DDPM
from modules.trainers.VDM import VDM
from modules.trainers.EDM import EDM
from modules.trainers.PFGMpp import PFGMpp
from modules.trainers.genmod import genmod






def main(args):
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ------------------------
    #  INIT DATAMODULE
    # ------------------------
    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    train_dataset = MNIST("/media/volume/sdb/data", train=True, download=True, transform=tf)
    val_dataset = MNIST("/media/volume/sdb/data", train=False, download=True, transform=tf)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True, 
                                               num_workers=args.num_workers, 
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=args.num_workers, 
                                             pin_memory=True)
    
    # Set float32 matrix multiplication precision to medium
    torch.set_float32_matmul_precision('medium')


    # ------------------------
    #  INIT MODEL
    # ------------------------
    model = DDPM()
    # model = VDM()
    # model = EDM()
    # model = PFGMpp()
    # model = genmod()
    

    # ------------------------
    #  INIT TRAINER
    # ------------------------
    wandb_logger = WandbLogger(project='genmod', 
                               log_model='all')

    checkpoint = ModelCheckpoint(monitor='train_loss',
                                 mode='min', 
                                 save_last=True)

    trainer = pl.Trainer(devices=args.gpu, 
                         accelerator='gpu',
                         precision=16,
                         callbacks=[checkpoint],
                         logger=wandb_logger, 
                         max_epochs=args.n_epochs, 
                         default_root_dir=save_dir, 
                         fast_dev_run=args.debug)

    # ------------------------
    #  Training
    # ------------------------

    trainer.fit(model, train_loader, val_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory Prediction')
    parser.add_argument('--save_dir', default='./results/training/DDPM',
                        type=str,help='path to directory for storing the checkpoints etc.')
    parser.add_argument('-b','--batch_size', default=256, type=int,
                        help='Batch size')
    parser.add_argument('-ep','--n_epochs', default=20, type=int,
                        help='Number of epochs')
    parser.add_argument('-g','--gpu', default=1, type=int,
                        help='num gpus')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='num workers for data module.')
    parser.add_argument('--debug', default=False, type=bool,
                        help='fast_dev_run argument')
    hparams = parser.parse_args()
    main(hparams)