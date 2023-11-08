import torch, os
import torch.nn.functional as F
import lightning as pl
from einops import rearrange
import torchgeometry as tgm

from torchvision.utils import save_image, make_grid

from modules.networks.UnetViT import UViT
from modules.networks.Unet import ContextUnet
from modules.utils.schedulers import linear_beta_schedule



class ColdDiff(pl.LightningModule):
    def __init__(self,
                n_T=1000,
                n_feat=128
                ):
        super(ColdDiff, self).__init__()
        self.nn_model = ContextUnet(in_channels=1, n_feat=n_feat)

        self.betas = linear_beta_schedule(n_T)
        
        self.gaussian_kernels = nn.ModuleList(self.get_kernels())

        self.n_T = n_T

    def forward(self, x, t):
        return self.nn_model(x_t, t / self.n_T)
    
    # blur(), get_conv(), get_kernels() are all subfunctions for sampling the forward ~diffusion~
    # process in Cold Diffusion
    def blur(self, dims, std):
        return tgm.image.get_gaussian_kernel2d(dims, std)

    def get_conv(self, dims, std, mode='circular'):
        kernel = self.blur(dims, std)
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=dims, padding=int((dims[0]-1)/2), padding_mode=mode,
                         bias=False, groups=self.channels)
        with torch.no_grad():
            kernel = torch.unsqueeze(kernel, 0)
            kernel = torch.unsqueeze(kernel, 0)
            kernel = kernel.repeat(self.channels, 1, 1, 1)
            conv.weight = nn.Parameter(kernel)

        return conv

    def get_kernels(self):
        kernels = []
        for i in range(self.n_T):
            kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std*(i+1), self.kernel_std*(i+1)) ) )

        return kernels

    def sample_forward_diffusion(self, x_0, _ts, noise):
        '''
        In cold diffusion the "forward diffusion process" is an abritrary image transformation. 
        For this example we use a basic image blurring operation, i.e., convolutions. 
        '''
        all_blurs = []
        x = x_0
        for i in range(self.n_T+1):
            with torch.no_grad():
                x = self.gaussian_kernels[i](x)
                all_blurs.append(x)

        all_blurs = torch.stack(all_blurs)

        choose_blur = []
        # step is batch size as well so for the 49th step take the step(batch_size)
        for step in range(t.shape[0]):
            if step != -1:
                choose_blur.append(all_blurs[t[step], step])
            else:
                choose_blur.append(x_0[step])

        return torch.stack(choose_blur)
    
    @torch.no_grad()
    def sample_loop(self, batch):
        '''
        Cold Diffusion Sampling - Algorithm 2 in paper: https://arxiv.org/abs/2208.09392
        x_t = x_T
        t = T
        for i in range(num_steps):
            x_0_hat = nn_model(x_t, t)
            x_t-1 = x_t - D(x_0_hat, t) + D(x_0_hat, t-1)
            x_t = x_t-1
        end
        D() is the degradation operator - in this basic case the self.gaussian_kernels function
        '''
        self.nn_model.eval()
        x_t = torch.randn_like(batch)    # generate initial random noise samples i.e. x_T
        for i in range(self.n_T, 1):
            x_0_hat = self.nn_model(x_T, (i / self.n_T).long())
            x_tm1 = x_t - self.gaussian_kernels[i](x_0_hat) + self.gaussian_kernels[i-1](x_0_hat)
            x_t = x_tm1
        
        return x_t
        
    
#     def gen_sample_2(self, batch_size=16, img=None, t=None, noise_level=0):

#         self.denoise_fn.eval()

#         if t == None:
#             t = self.num_timesteps

#         if self.blur_routine == 'Individual_Incremental':
#             img = self.gaussian_kernels[t - 1](img)

#         else:
#             for i in range(t):
#                 with torch.no_grad():
#                     img = self.gaussian_kernels[i](img)

#         orig_mean = torch.mean(img, [2, 3], keepdim=True)
#         print(orig_mean.squeeze()[0])

#         temp = img
#         if self.discrete:
#             img = torch.mean(img, [2, 3], keepdim=True)
#             img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])

#         noise = torch.randn_like(img) * noise_level
#         img = img + noise

#         # 3(2), 2(1), 1(0)
#         xt = img
#         direct_recons = None
#         while (t):
#             step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
#             x = self.denoise_fn(img, step)

#             if self.train_routine == 'Final':
#                 if direct_recons == None:
#                     direct_recons = x

#                 if self.sampling_routine == 'default':
#                     if self.blur_routine == 'Individual_Incremental':
#                         x = self.gaussian_kernels[t - 2](x)
#                     else:
#                         for i in range(t - 1):
#                             with torch.no_grad():
#                                 x = self.gaussian_kernels[i](x)

#                 elif self.sampling_routine == 'x0_step_down':
#                     x_times = x
#                     for i in range(t):
#                         with torch.no_grad():
#                             x_times = self.gaussian_kernels[i](x_times)
#                             if self.discrete:
#                                 if i == (self.num_timesteps - 1):
#                                     x_times = torch.mean(x_times, [2, 3], keepdim=True)
#                                     x_times = x_times.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])

#                     x_times_sub_1 = x
#                     for i in range(t - 1):
#                         with torch.no_grad():
#                             x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

#                     x = img - x_times + x_times_sub_1
#             img = x
#             t = t - 1

#         # img = img - noise

#         return xt, direct_recons, img
    
    def loss(self, x_0):
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x).long()  # t ~ Uniform(0, n_T)
        x_t = self.sample_forward_diffusion(x, _ts)
        return F.mse_loss(x_0, self.nn_model(x_t, _ts / self.n_T))

    def training_step(self, batch, batch_idx):
        images, _ = batch
        loss = self.loss(images)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        loss = self.loss(images)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        lr = 1e-4
        opt = torch.optim.Adam(self.nn_model.parameters(), lr=lr)
        return opt