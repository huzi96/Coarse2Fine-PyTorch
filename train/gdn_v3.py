# The implementation of GDN is inherited from
# https://github.com/jorge-pessoa/pytorch-gdn,
# under the MIT License.
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function
import numpy as np

class LowerBound(Function):
    """
    Low_bound make the numerical calculation close to the bound
    """
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y*torch.ones_like(x))
        x = torch.clamp(x, min=y)
        return x

    @staticmethod
    def backward(ctx, g):
        x, y = ctx.saved_tensors
        grad1 = g.clone()
        pass_through_if = torch.logical_or(x >= y, g < 0)
        t = pass_through_if
        return grad1*t, None

lower_bound = LowerBound.apply
class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
  
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.build(ch)
  
    def build(self, ch):
        self.pedestal_data = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset
  
        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal_data)
        self.beta = nn.Parameter(beta)
        self.register_parameter('beta', self.beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal_data
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.register_parameter('gamma', self.gamma)
        self.pedestal_tensor = torch.FloatTensor([self.pedestal_data])
        self.register_buffer('pedestal', self.pedestal_tensor)


    def forward(self, inputs):
        
        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        
        beta = lower_bound(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        
        gamma = lower_bound(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        # Apply norm
        outputs = inputs / norm_

        
        return outputs

class IGDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
  
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(IGDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)
  
    def build(self, ch):
        self.pedestal_data = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset
  
        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal_data)
        self.beta = nn.Parameter(beta)
        self.register_parameter('beta', self.beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal_data
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.register_parameter('gamma', self.gamma)
        self.pedestal_tensor = torch.FloatTensor([self.pedestal_data])
        self.register_buffer('pedestal', self.pedestal_tensor)

    def forward(self, inputs):

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        
        beta = lower_bound(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        
        gamma = lower_bound(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        
        outputs = inputs * norm_

        return outputs
