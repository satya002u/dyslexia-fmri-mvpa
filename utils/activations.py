import torch
import torch.nn as nn
import torch.nn.functional as F

class SSReLU(nn.Module):
    """Stochastic Shifted ReLU activation function for PyTorch."""
    
    def __init__(self, alpha=0.2, p=1.0):
        super(SSReLU, self).__init__()
        self.alpha = alpha
        self.p = p
    
    def forward(self, x):
        if self.training:
            # noise_std = self.alpha * torch.abs(x) ** self.p
            noise_std = self.alpha * (torch.abs(x) + 1e-6) ** self.p
            noise = torch.randn_like(x) * noise_std
            return torch.maximum(torch.zeros_like(x), x + noise)
        else:
            return F.relu(x)


class SSGeLU(nn.Module):
    """Stochastic Shifted GELU activation function for PyTorch."""
    
    def __init__(self, alpha=0.2, p=1.0):
        super(SSGeLU, self).__init__()
        self.alpha = alpha
        self.p = p
    
    def forward(self, x):
        if self.training:
            # noise_std = self.alpha * torch.abs(x) ** self.p
            noise_std = self.alpha * (torch.abs(x) + 1e-6) ** self.p
            noise = torch.randn_like(x) * noise_std
            return F.gelu(x + noise)
        else:
            return F.gelu(x)
