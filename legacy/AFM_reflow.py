import torch
import torch.nn as nn
import torch.nn.functional as F


def get_AFM_data(upsrt_encoder, batch, target):
    '''
    Take of batch of Proteins with n unique views and get Rectified Flow data.
    
    upsrt_encoder: Pretrained UpSRT Encoder.
    batch: ((B, N, C, H, W), Q) tuple of tensor where N is list of unique 
                views for single protein and corresponding view information.
    target: ((B, C, H, W), Q) tuple of tensors of unique view for single 
                protein and corresponding view information.
    
    t: (B,) tensor of timesteps.
    xt: (B, C, H, W) tensor of linearly interpolated tensor between embedding
                matrix and target query view.
    velocity: (B, C, H, W) tensor of line direction between embedding matrix 
                and target query view.
    
    '''
    t = torch.rand((batch.shape[0],))
    
    x0 = upsrt_encoder(batch)
    x1 = target[0]
    
    xt = t * x1 + (1 - t) * x0
    velocity = x1 - x0
    
    return xt, t, velocity

def compute_Rectified_Flow_loss(model, target, xt, t, velocity):
    '''
    Compute Rectified Flow loss.
    
    model: UNetModel
    target: ((B, C, H, W), Q) tuple of tensors of unique view for single protein 
                    and corresponding view information.
    xt: (B, C, H, W) tensor of linearly interpolated tensor between embedding matrix 
                    and target query view.
    t: (B,) tensor of timesteps.
    velocity: (B, C, H, W) tensor of line direction between embedding matrix and 
                    target query view.
    
    returns: () loss value
    '''
    velocity_hat = model(xt, t, target[1])
    
    loss = F.mse_loss(velocity_hat, velocity)
    
    return loss

def generate_AFM_queried_view(model, upsrt_encoder, batch, target_query_angle, num_timesteps):
    '''
    Take of batch of Proteins with n unique views and get new view at query angle.

    model: UNetModel
    upsrt_encoder: Pretrained UpSRT Encoder.
    batch: ((B, N, C, H, W), Q) tuple of tensor where N is list of unique views for 
                    single protein and corresponding view information.
    target_query_angle: (B, ) tensor of query view angle.
    
    x: (B, C, H, W) tensor of new view at target query angle.
    '''
    x = upsrt_encoder(batch)
    dt = 1. / num_timesteps
    
    for i in range(num_timesteps):
        t = torch.ones((x.shape[0],)) * i * dt
        velocity = model(x, t, target_query_angle)
        x = x + velocity * dt
    
    return x


    
    