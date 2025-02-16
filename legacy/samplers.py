import torch
import torch.nn as nn



@torch.no_grad()
def sample_ode(self, z0=None, N=None):
    if N is None:
        N = self.N
    dt = 1. / N
    trajectory = []
    z = z0.detach().clone().to(self.device)
    
    trajectory.append(z.detach().clone())
    for i in range(N):
        t = torch.ones((z.shape[0],)) * i / N
        t = t.to(self.device)
        # Initial prediction using current state
        pred = self.model(z, t)
        # Tentative next state using the initial prediction
        z_tentative = z + pred * dt
        
        # Calculate t for the next time step
        t_next = torch.ones((z.shape[0],)) * (i + 1) / N
        t_next = t_next.to(self.device)
        # Prediction at the tentative next state
        pred_next = self.model(z_tentative, t_next)
        
        # Correct the next state estimate using the average of the initial and new predictions
        z = z + (pred + pred_next) * 0.5 * dt
        
        trajectory.append(z.detach().clone())
    return trajectory




@torch.no_grad()
def sample_ode(self, z0=None, N=None):
    if N is None:
        N = self.N
    dt = 1. / N
    trajectory = []
    z = z0.detach().clone().to(self.device)
    
    trajectory.append(z.detach().clone())
    for i in range(N):
        t_current = torch.ones((z.shape[0],)) * i / N
        t_current = t_current.to(self.device)
        
        # k1 is the derivative at the start of the interval
        k1 = self.model(z, t_current)
        
        # k2 is the derivative at the midpoint, using k1 to update z
        z_k2 = z + 0.5 * k1 * dt
        t_mid = torch.ones((z.shape[0],)) * (i + 0.5) / N
        t_mid = t_mid.to(self.device)
        k2 = self.model(z_k2, t_mid)
        
        # k3 is also at the midpoint, but using k2 to update z
        z_k3 = z + 0.5 * k2 * dt
        k3 = self.model(z_k3, t_mid)
        
        # k4 is the derivative at the end of the interval, using k3 to update z
        z_k4 = z + k3 * dt
        t_next = torch.ones((z.shape[0],)) * (i + 1) / N
        t_next = t_next.to(self.device)
        k4 = self.model(z_k4, t_next)
        
        # Combine k1, k2, k3, k4 to get the next z
        z = z + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
        
        trajectory.append(z.detach().clone())
    return trajectory


@torch.no_grad()
def sample_ode_corrected(self, z0=None, N=None):
    if N is None:
        N = self.N
    dt = 1. / N
    trajectory = []
    z = z0.detach().clone().to(self.device)
    
    trajectory.append(z.detach().clone())
    for i in range(N):
        t = torch.ones((z.shape[0],)) * i / N
        t = t.to(self.device)
        pred = self.model(z, t)
        
        # Initial Euler step
        z_pred = z + pred * dt
        
        # Evaluate correction term (conceptually using second derivative or a better estimate of the first derivative)
        t_next = torch.ones((z.shape[0],)) * (i + 1) / N
        t_next = t_next.to(self.device)
        pred_next = self.model(z_pred, t_next)
        
        # Apply the correction term
        correction = (pred_next - pred) * (dt ** 2) / 2  # Simplistic second-order correction
        z = z + pred * dt + correction
        
        trajectory.append(z.detach().clone())
    return trajectory


@torch.no_grad()
def sample_ode_abm(self, z0=None, N=None, dt=None):
    if N is None:
        N = self.N
    if dt is None:
        dt = 1. / N
    trajectory = [z0.detach().clone().to(self.device)]

    # Assuming f_history is accessible and stores at least the last two f(t, y) evaluations
    z = z0.detach().clone().to(self.device)
    f_history = [self.model(z, torch.zeros(1, device=self.device))]

    for i in range(1, N):
        t = torch.ones((z.shape[0],), device=self.device) * i / N
        # Predictor (Adams-Bashforth 2-step)
        f_current = self.model(z, t)
        f_history.append(f_current)
        if len(f_history) > 2:  # Ensure we have at least two steps of history
            z_pred = z + dt / 2 * (3 * f_history[-1] - f_history[-2])

            # Corrector (Adams-Moulton 2-step)
            t_next = torch.ones((z.shape[0],), device=self.device) * (i + 1) / N
            f_next_pred = self.model(z_pred, t_next)
            z = z + dt / 2 * (f_history[-1] + f_next_pred)

            trajectory.append(z.detach().clone())
            # Update history
            f_history.append(f_next_pred)
            if len(f_history) > 2:
                f_history.pop(0)  # Keep the last two only

    return trajectory
