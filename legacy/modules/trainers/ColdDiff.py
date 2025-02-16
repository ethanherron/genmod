import torch, os
import torch.nn.functional as F
import lightning as pl
from einops import rearrange
import torchgeometry as tgm

from torchvision.utils import save_image, make_grid

from modules.networks.UnetViT import UViT
from modules.networks.Unet import ContextUnet
from modules.utils.schedulers import linear_beta_schedule

from modules.trainers.base import BaseDiffusionModule




class ColdDiff(BaseDiffusionModule):
    def __init__(self,
                number_of_timesteps=1000
                ):
        super().__init__(number_of_timesteps)
        self.betas = linear_beta_schedule(n_T)
        self.gaussian_kernels = nn.ModuleList(self.get_kernels())

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
        '''
        There are multiple different options in ColdDiff, but we will use the 
        'Incremental' version for this implementation.
        '''
        kernels = []
        for i in range(self.n_T):
            kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std*(i+1), self.kernel_std*(i+1)) ) )

        return kernels

    def forward_diffusion_process(self, x_0, _ts):
        '''
        In cold diffusion the "forward diffusion process" is an abritrary image transformation. 
        For this example we use a basic image blurring operation by applying convolutions. 
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
    def reverse_diffusion_process(self, batch):
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
        for i in range(self.n_T):
            with torch.no_grad():
                x_t = self.gaussian_kernels[i](img)
        for i in range(self.n_T, 1):
            x_0_hat = self.nn_model(x_T, (i / self.n_T).long())
            x_tm1 = x_t - self.gaussian_kernels[i](x_0_hat) + self.gaussian_kernels[i-1](x_0_hat)
            x_t = x_tm1
        
        return x_t
    
    def reverse_diffusion_step(self)
    
    def loss(self, x_0):
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x).long()  # t ~ Uniform(0, n_T)
        x_t = self.sample_forward_diffusion(x, _ts)
        return F.mse_loss(x_0, self.nn_model(x_t, _ts / self.n_T))