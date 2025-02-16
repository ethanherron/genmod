import torch, os
import torch.nn.functional as F
from einops import rearrange
from modules.trainers.base import BaseDiffusionModule
from ema_pytorch import EMA

# helper functions
def pad(var):
    if var.shape == ():
        return rearrange(var, ' -> 1 1 1 1')
    else:
        return rearrange(var, 'b -> b 1 1 1')
    
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))
    
    
    

class CMIsolation(BaseDiffusionModule):
    def __init__(self,
                 number_of_timesteps=50,
                n_feat=128,
                sigma_min = 0.002,     # min noise level
                sigma_max = 80,        # max noise level
                sigma_data = 0.302,    # standard deviation of data distribution (MNIST)
                rho = 7,               # controls the sampling schedule
                P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
                P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
                S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in paper
                S_tmin = 0.05,
                S_tmax = 50.,
                S_noise = 1.003,
                ):
        super(CMIsolation, self).__init__(number_of_timesteps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        self.target_nn_model = EMA(self.nn_model,
                                   beta=0.999,
                                   update_after_step=1,
                                   update_every=5
                                  )
        
    def preconditioners(self, sigmas):
        c_in = 1 * (sigmas ** 2 + self.sigma_data ** 2) ** -0.5
        c_noise = log(sigmas) * 0.25
        c_skip = (self.sigma_data ** 2) / (sigmas ** 2 + self.sigma_data ** 2)
        c_out = sigmas * self.sigma_data * (self.sigma_data ** 2 + sigmas ** 2) ** -0.5
        return c_in, c_noise, c_skip, c_out
    
    def loss_weight(self, sigmas):
        sigmas = rearrange(sigmas, 'b 1 1 1 -> b')
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2
    
    def sample_sigmas(self, x_0):
        inv_rho = 1 / self.rho
        
        steps = torch.randint(1, self.n_T-1, (x_0.shape[0],)).to(x_0).long()  # t ~ Uniform(0, n_T)
        sigma_t = (self.sigma_max ** inv_rho + steps / (self.n_T - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
        
        sigma_tp1 = (self.sigma_max ** inv_rho + (steps + 1) / (self.n_T - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
        
        return pad(sigma_t), pad(sigma_tp1)
    
    def forward(self, model, x_t, sigma):
        c_in, c_noise, c_skip, c_out = self.preconditioners(sigma)
        network_output = model((c_in * x_t), c_noise)
        x_0_hat = c_skip * x_t + c_out * network_output
        return x_0_hat.clamp(0.,1.)
    
    @torch.no_grad()
    def euler_solver(self, x_t, sigma_t, sigma_tp1, x_0):
        d = (x_t - x_0) / sigma_t
        # x_t_minus_1 = x_t + d * d/dt
        x_tp1 = x_t + d * (sigma_tp1 - sigma_t)
        return x_tp1.detach()
    
    def loss(self, x_0):
        sigma_t, sigma_tp1 = self.sample_sigmas(x_0)
        x_t = self.forward_diffusion_process(x_0, sigma_t)
        
        # pred
        pred = self.forward(self.nn_model, x_t, sigma_t)
        
        # target
        x_tp1 = self.euler_solver(x_t, sigma_t, sigma_tp1, x_0)
        target = self.forward(self.target_nn_model, x_tp1, sigma_tp1)
        
        reconstruction_loss = F.mse_loss(pred, target, reduction='none').mean(dim=(1,2,3))
        # print('pred shape: ', pred.shape)
        # print('target shape: ', target.shape)
        # print('recon loss shape: ', reconstruction_loss.shape)
        # print('loss weights shape: ', self.loss_weight(sigma_t).shape)
        # print(' ')
        # exit()
        loss = (self.loss_weight(sigma_t) * reconstruction_loss).mean()
        return loss
    
    def forward_diffusion_process(self, x_0, sigmas):
        eps = torch.randn_like(x_0).to(x_0)
        x_t = x_0 + sigmas * eps
        return x_t
    
    @torch.no_grad()
    def reverse_diffusion_process(self, x_t):
        sigmas = self.sample_schedule()
        for i in range(len(sigmas)-1):
            sigma_t, sigma_tm1 = sigmas[i], sigmas[i+1]
            x_t = self.reverse_diffusion_step(x_t, sigma)
    
    @torch.no_grad()
    def reverse_diffusion_step(self, x_t, sigma_t, sigma_tm1):
        output_tm1 = self.forward(x_t, sigma_t)
        d = (x_t - output_tm1) / sigma_t
        dt = sigma_tm1 - sigma
        x_tm1 = x_t + d * dt
        return x_tm1
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.target_nn_model.update()
        
        