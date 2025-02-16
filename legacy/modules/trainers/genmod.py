import torch, os
import torch.nn.functional as F
import lightning as pl
from einops import rearrange

from torchvision.utils import save_image, make_grid

from modules.networks.UnetViT import UViT
from modules.networks.Unet import ContextUnet, ContextUnet_genmod
from modules.utils import linear_beta_schedule



class genmod(pl.LightningModule):
    def __init__(self,
                n_T=500,
                n_feat=128
                ):
        super(genmod, self).__init__()
        # self.nn_model = UViT()
        self.nn_model = ContextUnet_genmod(in_channels=1, out_channels=2, n_feat=n_feat)

        self.betas = linear_beta_schedule(n_T)

        self.ddpm_schedules = self.register_ddpm_schedules(self.betas)

        for k, v in self.ddpm_schedules.items():
            self.register_buffer(k, v)

        self.n_T = n_T

    def register_ddpm_schedules(self, beta_t):
        """
        Returns pre-computed schedules for DDPM sampling, training process.
        """

        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def forward(self, x):
        # randomly sample noise
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        
        # randomly sample time steps for input noisy image and target noisy image
        rand_t_1 = torch.randint(0, self.n_T, (x.shape[0],)).to(x).long()  # t ~ Uniform(0, n_T)
        rand_t_2 = torch.randint(0, self.n_T, (x.shape[0],)).to(x).long()  # t ~ Uniform(0, n_T)
        # sort randomly sampled times so t_input >= t_ouput (we always want to learn reverse diffusion OR identity function)
        t_input = torch.maximum(rand_t_1, rand_t_2)
        t_output = torch.minimum(rand_t_1, rand_t_2)
        
        # compute input noisy image and target noisy image
        x_input = self.sample_forward_diffusion(x, t_input, noise)
        x_output_target = self.sample_forward_diffusion(x, t_output, noise)
        
        # predict denoised image
        prediction = self.nn_model(x_input, t_input / self.n_T, t_output / self.n_T)
        
        # stack x_0 and eps
        target = torch.cat((noise, x), dim=1)
        
        # compute reconstruction loss mse([x_output_prediction, eps_prediction], [x_output_target, eps_target])
        return F.mse_loss(prediction, target)
        

    def sample_forward_diffusion(self, x, _ts, noise):
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )
        return x_t

    def sample_loop(self, batch):
        x_i = torch.randn_like(batch).to(batch)  # x_T ~ N(0, 1), sample initial noise
        for i in range(self.n_T, 1, -1):
            t_input = torch.tensor([i / self.n_T]).to(batch)
            t_input = t_input.repeat(batch.size(0),1,1,1)
            t_output = torch.tensor([i-1 / self.n_T]).to(batch)
            t_output = t_output.repeat(batch.size(0),1,1,1)
            
            x_i = self.nn_model(x_i, t_input, t_output)
        return x_i

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