import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Dict, Callable
import numpy as np


class ODESampler:
    """
    ODE sampler for flow matching generation.
    Supports multiple solvers and conditional generation with classifier-free guidance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50,
        solver: str = 'euler',
        device: str = 'cuda'
    ):
        """
        Args:
            model: FlowMatchingModel instance
            num_steps: Number of ODE integration steps
            solver: ODE solver ('euler', 'midpoint', 'rk4')
            device: Device to run on
        """
        self.model = model
        self.num_steps = num_steps
        self.solver = solver
        self.device = device
        
        self.solver_fn = self._get_solver()
    
    def _get_solver(self) -> Callable:
        """Get the ODE solver function"""
        solvers = {
            'euler': self._euler_step,
            'midpoint': self._midpoint_step,
            'rk4': self._rk4_step
        }
        
        if self.solver not in solvers:
            raise ValueError(f"Unknown solver: {self.solver}. "
                           f"Available: {list(solvers.keys())}")
        
        return solvers[self.solver]
    
    def _euler_step(
        self,
        z_t: torch.Tensor,
        t: float,
        dt: float,
        mask: torch.Tensor,
        conditions: Optional[Dict] = None
    ) -> torch.Tensor:
        """Euler method: z_{t+dt} = z_t + v_t * dt"""
        s = torch.ones(z_t.shape[0], device=self.device) * t
        v_t = self.model(z_t, s, mask, conditions)
        return z_t + v_t * dt
    
    def _midpoint_step(
        self,
        z_t: torch.Tensor,
        t: float,
        dt: float,
        mask: torch.Tensor,
        conditions: Optional[Dict] = None
    ) -> torch.Tensor:
        """Midpoint method (2nd order)"""
        s_t = torch.ones(z_t.shape[0], device=self.device) * t
        s_mid = torch.ones(z_t.shape[0], device=self.device) * (t + dt/2)
        
        v_t = self.model(z_t, s_t, mask, conditions)
        z_mid = z_t + v_t * (dt/2)
        
        v_mid = self.model(z_mid, s_mid, mask, conditions)
        return z_t + v_mid * dt
    
    def _rk4_step(
        self,
        z_t: torch.Tensor,
        t: float,
        dt: float,
        mask: torch.Tensor,
        conditions: Optional[Dict] = None
    ) -> torch.Tensor:
        """Runge-Kutta 4th order method"""
        B = z_t.shape[0]
        
        s_t = torch.ones(B, device=self.device) * t
        k1 = self.model(z_t, s_t, mask, conditions)
        
        s_mid1 = torch.ones(B, device=self.device) * (t + dt/2)
        k2 = self.model(z_t + k1 * dt/2, s_mid1, mask, conditions)
        
        s_mid2 = torch.ones(B, device=self.device) * (t + dt/2)
        k3 = self.model(z_t + k2 * dt/2, s_mid2, mask, conditions)
        
        s_end = torch.ones(B, device=self.device) * (t + dt)
        k4 = self.model(z_t + k3 * dt, s_end, mask, conditions)
        
        return z_t + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples using ODE solver.
        
        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length
            conditions: Optional conditioning information
            guidance_scale: Classifier-free guidance scale (>1 for stronger conditioning)
            mask: Optional mask for variable-length sequences
            show_progress: Whether to show progress bar
        
        Returns:
            z_1: (B, L, D) generated joint states at t=1
        """
        self.model.eval()
        
        d_joint = self.model.d_joint
        z_t = torch.randn(batch_size, seq_len, d_joint, device=self.device)
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(self.device)
        
        dt = 1.0 / self.num_steps
        
        iterator = range(self.num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc='Sampling', leave=False)
        
        for step in iterator:
            t = step / self.num_steps
            
            if guidance_scale != 1.0 and conditions is not None:
                v_cond = self._get_velocity(z_t, t, mask, conditions)
                v_uncond = self._get_velocity(z_t, t, mask, None)
                
                v_t = v_uncond + guidance_scale * (v_cond - v_uncond)
                
                s = torch.ones(batch_size, device=self.device) * t
                z_t = z_t + v_t * dt
            else:
                z_t = self.solver_fn(z_t, t, dt, mask, conditions)
        
        return z_t
    
    def _get_velocity(
        self,
        z_t: torch.Tensor,
        t: float,
        mask: torch.Tensor,
        conditions: Optional[Dict]
    ) -> torch.Tensor:
        """Helper to get velocity prediction"""
        s = torch.ones(z_t.shape[0], device=self.device) * t
        return self.model(z_t, s, mask, conditions)
    
    @torch.no_grad()
    def sample_with_trajectory(
        self,
        batch_size: int,
        seq_len: int,
        save_every: int = 10,
        conditions: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Generate samples and save intermediate states.
        Useful for visualization and debugging.
        
        Returns:
            z_1: Final samples
            trajectory: List of intermediate states
        """
        self.model.eval()
        
        d_joint = self.model.d_joint
        z_t = torch.randn(batch_size, seq_len, d_joint, device=self.device)
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        trajectory = [z_t.cpu().clone()]
        dt = 1.0 / self.num_steps
        
        for step in tqdm(range(self.num_steps), desc='Sampling with trajectory'):
            t = step / self.num_steps
            z_t = self.solver_fn(z_t, t, dt, mask, conditions)
            
            if step % save_every == 0 or step == self.num_steps - 1:
                trajectory.append(z_t.cpu().clone())
        
        return z_t, trajectory


class AdaptiveSampler(ODESampler):
    """
    Adaptive ODE sampler with dynamic step size.
    Uses error estimation to adjust step size.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50,
        device: str = 'cuda',
        rtol: float = 1e-3,
        atol: float = 1e-5
    ):
        super().__init__(model, num_steps, 'rk4', device)
        self.rtol = rtol
        self.atol = atol
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        conditions: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None,
        show_progress: bool = True
    ) -> torch.Tensor:
        """Adaptive sampling with error control"""
        self.model.eval()
        
        d_joint = self.model.d_joint
        z_t = torch.randn(batch_size, seq_len, d_joint, device=self.device)
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        t = 0.0
        dt = 1.0 / self.num_steps
        
        pbar = tqdm(total=1.0, desc='Adaptive sampling') if show_progress else None
        
        while t < 1.0:
            if t + dt > 1.0:
                dt = 1.0 - t
            
            z_full = self._rk4_step(z_t, t, dt, mask, conditions)
            
            z_half1 = self._rk4_step(z_t, t, dt/2, mask, conditions)
            z_half2 = self._rk4_step(z_half1, t + dt/2, dt/2, mask, conditions)
            
            error = torch.abs(z_full - z_half2).max()
            
            tolerance = self.atol + self.rtol * torch.abs(z_t).max()
            
            if error < tolerance:
                z_t = z_half2
                t += dt
                
                if pbar is not None:
                    pbar.update(dt)
                
                dt = min(dt * 1.5, 1.0 - t)
            else:
                dt = dt * 0.5
        
        if pbar is not None:
            pbar.close()
        
        return z_t