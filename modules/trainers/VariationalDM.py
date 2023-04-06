import torch, os
import torch.nn.functional as F
import lightning as pl
from einops import rearrange

from torchvision.utils import save_image, make_grid

from modules.networks.Unet import ContextUnet

class VDM(pl.LightningModule):
    def __init__(self,
                n_T=500,
                n_feat=128
                ):
        super(VDM, self).__init__()
        self.nn_model = ContextUnet(in_channels=1, n_feat=n_feat)

        self.n_T = n_T

    def forward(self, x):
        return self.nn_model(x)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        loss = self.forward(images)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        loss = self.forward(images)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        lr = 1e-4
        opt = torch.optim.Adam(self.nn_model.parameters(), lr=lr)
        return opt