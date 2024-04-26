import torch
import torch.nn as nn
import torch.nn.functional as F




def get_CH_data(batch):
    '''
    Get Cahn-Hilliard data from batch of trajectories.
    
    batch: (B, N, C, H, W) tensor where N is list of trajectories for single morphology
    
    n: (B, ) tensor of timesteps. Point in CH solver trajectory. 
    t: (B, ) tensor of timesteps. Point in linear interpolation between morphologies n and n+1.
    xt: (B, C, H, W) tensor of linearly interpolated morphology from n to n+1.
    target: (B, C, H, W) tensor of line direction between morphologies n and n+1.
    
    returns: (B, T, C, H, W) tensor where T is tuple of timesteps (t, t+1) for each morphology
    '''
    n = torch.randint(low=0, high=batch.shape[1], size=(batch.shape[0],))
    x0 = batch[:, n, :, :, :]
    x1 = batch[:, n + 1, :, :, :]
    
    t = torch.rand((batch.shape[0],))
    
    xt = t * x1 + (1 - t) * x0
    
    target = x1 - x0
    
    return n, t, xt, target

def compute_CH_loss(model, n, t, xt, target):
    '''
    Compute Cahn-Hilliard Solver loss.
    
    model: UNetModel
    n: (B, ) tensor of timesteps. Point in CH solver trajectory.
    t: (B, ) tensor of timesteps. Point in linear interpolation between morphologies n and n+1.
    xt: (B, C, H, W) tensor of linearly interpolated morphology from n to n+1.
    target: (B, C, H, W) tensor of line direction between morphologies n and n+1.
    
    returns: () loss value
    '''
    pred = model(xt, n, t)
    
    loss = F.mse_loss(pred, target)
    
    return loss

def generate_CH_morphology(model, num_timesteps, nfe):
    '''
    Generate morphology using trained model.
    
    model: UNetModel
    num_timesteps: number of timesteps in CH solver trajectory (how many intermediate morphologies to generate)
    nfe: number of forward euler steps in between each intermediate morphology
    
    x: [x_n] tensor of shape (B, C, H, W) where x_n is the nth morphology in the trajectory.
    '''
    x = torch.randn((1, 1, 64, 64))
    dt = 1. / nfe
    trajectory = []
    
    for n in range(num_timesteps):
        for i in range(nfe):
            t = torch.ones((1,)) * (i * dt)
            velocity = model(x, n, t)
            x = x + velocity * dt
        trajectory.append(x)
    
    return trajectory
    
    