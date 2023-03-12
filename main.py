import argparse
import torch
from data import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from trainer import *






def main(args):
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        
    
    # ------------------------
    # 0 INIT DATAMODULE
    # ------------------------
    train_dataset = SMIDataset('train')
    val_dataset = SMIDataset('val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    # ------------------------
    # 1 INIT MODEL
    # ------------------------

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger(save_dir, 
                                          name="tensorboard_logs")

    csv_logger = pl.loggers.CSVLogger(logger.save_dir, 
                                      name=logger.name, 
                                      version=logger.version)

    checkpoint = ModelCheckpoint(monitor='train_loss',
                                 dirpath=logger.log_dir, 
                                 filename='{epoch}-{step}',
                                 mode='min', 
                                 save_last=True)

    trainer = pl.Trainer(devices=args.gpu, 
                         accelerator='gpu', 
                         strategy='ddp',
                         callbacks=[checkpoint],
                         logger=[logger,csv_logger], 
                         max_epochs=args.n_epochs, 
                         default_root_dir=save_dir, 
                         fast_dev_run=args.debug)

    # ------------------------
    # 4 Training
    # ------------------------

    trainer.fit(model, train_loader, val_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory Prediction')
    parser.add_argument('--save_dir', default='.results/training/debug',
                        type=str,help='path to directory for storing the checkpoints etc.')
    parser.add_argument('-b','--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('-ep','--n_epochs', default=50, type=int,
                        help='Number of epochs')
    parser.add_argument('-g','--gpu', default=2, type=int,
                        help='num gpus')
    parser.add_argument('--num_workers', default=24, type=int,
                        help='num workers for data module.')
    parser.add_argument('--debug', default=False, type=bool,
                        help='fast_dev_run argument')
    hparams = parser.parse_args()
    main(hparams)