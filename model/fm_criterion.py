import torch
import torch.nn.functional as F

class FlowMatchingEngine:
    @staticmethod
    def train_loss(model, x_1, conds):
        """
        x_1: Real 9-dimensional spatiotemporal matrix (B, 243, 9)
        """
        device = x_1.device
        x_0 = torch.randn_like(x_1) # Gaussian noise starting point
        t = torch.rand(x_1.shape[0], 1, 1, device=device) # Flow time step
        
        # Construct linear interpolation: x_t = (1-t)x_0 + t*x_1
        x_t = (1 - t) * x_0 + t * x_1
        target_v = x_1 - x_0 # Ideal velocity
        
        # Predict vector field
        pred_v = model(x_t, t.squeeze(), conds)
        return F.mse_loss(pred_v, target_v)

    @staticmethod
    @torch.no_grad()
    def sample(model, batch_size, conds, steps=50):
        """Use Euler method to solve ODE for inference"""
        device = next(model.parameters()).device
        x = torch.randn(batch_size, 243, 9, device=device)
        dt = 1.0 / steps
        
        for i in range(steps):
            t = torch.ones(batch_size, 1, device=device) * (i / steps)
            v = model(x, t, conds)
            x = x + v * dt
        return x