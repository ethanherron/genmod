import torch, os
import torch.nn.functional as F
import lightning as pl
from einops import rearrange

from torchvision.utils import save_image, make_grid

from modules.networks.UnetViT import UViT
from modules.networks.Unet import ContextUnet


# helper functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1



class BaseDiffusionModule(pl.LightningModule):
    '''
    This is the base diffusion model class.
    It is just the boilerplate training functions for lightning.
    The goal is for clarity across the different types of diffusion models
    so each other file will only have its bare differentiating factors.
    
    So each model will have its own:
    forward pass
    forward diffusion process (q(x_t\|x_{t-1}))
    reverse diffusion step    (p_{\theta}(x_{t-1}\|x_t))
    reverse diffusion process (dx = -\dot{\sigma}(t) \sigma(t) \nabla_x log p(x;\sigma(t))dt)
    loss
    '''
    def __init__(self,
                 number_of_timesteps,
                ):
        super().__init__()
        self.nn_model = ContextUnet(in_channels=1, n_feat=128)
        self.n_T = number_of_timesteps
        
    def forward(self, inputs):
        pass
    
    def loss(self, inputs):
        pass
    
    def forward_diffusion_process(self, inputs):
        pass
    
    @torch.no_grad()
    def reverse_diffusion_step(self, inputs):
        pass
    
    @torch.no_grad()
    def reverse_diffusion_process(self, inputs):
        pass
        
    def training_step(self, batch, batch_idx):
        images, _ = batch
        # images = normalize_to_neg_one_to_one(images)
        loss = self.loss(images)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        # images = normalize_to_neg_one_to_one(images)
        loss = self.loss(images)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        lr = 1e-4
        opt = torch.optim.Adam(self.nn_model.parameters(), lr=lr)
        return opt