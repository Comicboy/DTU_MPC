import numpy as np

from abc import ABC, abstractmethod


class DisturbanceModel(ABC):
    @abstractmethod
    def ffun(self, t, x, u, d, p):
        """Drift function f(t, x, u, d, p)"""
        pass

    @abstractmethod
    def gfun(self, t, x, u, d, p):
        """Diffusion function g(t, x, u, d, p)"""
        pass


class OU_DisturbanceModel(DisturbanceModel):
    '''
    Ornstein–Uhlenbeck (OU) process:
        dX_t = theta * (mu - X_t) dt + sigma dB_t

    Interpretation:
    - theta : Mean reversion rate. Higher = faster pull toward mu.
    - mu    : Long-term mean level that the process reverts to.
    - sigma : Noise intensity (diffusion). Controls smoothness/variance.
    '''
    
    model_type = "OU"
    
    def __init__(self, theta=0.1, mu=100.0, sigma=1.0):
        self.theta = theta      # mean reversion speed
        self.mu = mu            # long-term equilibrium level
        self.sigma = sigma      # volatility / noise strength

    def ffun(self, t, d):
        """OU drift: pulls X_t toward mu."""
        return self.theta * (self.mu - d)

    def gfun(self, t, d):
        """OU diffusion: constant noise strength."""
        return self.sigma



class BrownianMotion(DisturbanceModel):
    '''
    Standard Brownian Motion (Wiener process):
        dX_t = sigma dB_t

    Interpretation:
    - sigma : Diffusivity (noise scale). Larger sigma = rougher paths.
    - No drift term → process is a random walk around initial condition.
    '''
    
    model_type = "BM"
    
    def __init__(self, sigma=1.0, alpha = 100):
        self.sigma = sigma      # diffusion intensity
        self.alpha = alpha

    def ffun(self, t, d):
        """Zero drift: pure diffusion."""
        return 0

    def gfun(self, t, d):
        """Constant diffusion."""
        return self.sigma



class GeometricBrownianMotion(DisturbanceModel):
    ''' 
    Geometric Brownian Motion (GBM):
        dX_t = mu * X_t dt + sigma * X_t dB_t

    Interpretation:
    - mu    : Growth rate (drift). Positive = exponential growth.
    - sigma : Volatility multiplier. Noise scales with X_t → multiplicative.
    - X_t   : Always stays positive if initialized positive.
    '''
    
    model_type = "GBM"
    
    def __init__(self, mu=0.05, sigma=0.1):
        self.mu = mu            # exponential drift rate
        self.sigma = sigma      # multiplicative noise intensity

    def ffun(self, t, d):
        """GBM drift: proportional to the current value."""
        return self.mu * d

    def gfun(self, t, d):
        """GBM diffusion: multiplicative noise (sigma * X_t)."""
        return self.sigma * d   
class CoxIngersollRoss(DisturbanceModel):
    ''' 
    Cox–Ingersoll–Ross (CIR) process:
        dX_t = lambd * (xi - X_t) dt + gamma * sqrt(X_t) dB_t

    Interpretation:
    - lambd : Mean reversion speed (pulls X_t toward xi).
    - xi    : Long-term mean level (equilibrium).
    - gamma : Volatility coefficient; noise increases with sqrt(X_t).
    - Always stays non-negative.
    '''
    
    model_type = "CIR"
    
    def __init__(self, lambd=0.5, xi=100.0, gamma=1.0):
        self.lambd = lambd      # reversion speed
        self.xi = xi            # long-term mean level
        self.gamma = gamma      # diffusion scaling factor

    def ffun(self, t, d):
        """CIR drift: mean-reverting toward xi."""
        return self.lambd * (self.xi - d)

    def gfun(self, t, d):
        """CIR diffusion: gamma * sqrt(X_t), ensures non-negativity."""
        sqrt_d = np.sqrt(np.maximum(d, 0.0))
        return self.gamma * sqrt_d