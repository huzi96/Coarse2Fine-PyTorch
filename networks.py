# The implementation of GDN is inherited from
# https://github.com/jorge-pessoa/pytorch-gdn,
# under the MIT License.
#
# This file is being made available under the BSD License.  
# Copyright (c) 2021 Yueyu Hu
import argparse
import glob

import numpy as np
import torch
import torchvision as tv
from torch import nn, optim
import torch.nn.functional as F


import pickle
from PIL import Image
from PIL import ImageFilter
from torch.autograd import Function
import time
import os

from gdn_v2 import GDN
import subprocess as sp
import math

from translate_weights import *

# These are floating point bit masks used in AE
# They are currently for testing
# Setting them to 1 will make little differences
mask_a = 1
mask_b = 1

import tqdm
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = None

class Preprocess(object):
    def __init__(self):
        pass

    def __call__(self, PIL_img):
        img = np.asarray(PIL_img, dtype=np.float32)
        img /= 127.5
        img -= 1.0
        return img.transpose((2, 0, 1))
    
def quantize_image(img):
    img = torch.clamp(img, -1, 1)
    img += 1
    img = torch.round(img)
    img = img.to(torch.uint8)
    return img

class analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters):
        super(analysisTransformModel, self).__init__()
        self.t0 = nn.Sequential(
        )
        self.transform = nn.Sequential(
          nn.ZeroPad2d((1,2,1,2)),
          nn.Conv2d(in_dim, num_filters[0], 5, 2, 0),
          GDN(num_filters[0]),
          
          nn.ZeroPad2d((1,2,1,2)),
          nn.Conv2d(num_filters[0], num_filters[1], 5, 2, 0),
          GDN(num_filters[1]),
          
          nn.ZeroPad2d((1,2,1,2)),
          nn.Conv2d(num_filters[1], num_filters[2], 5, 2, 0),
          GDN(num_filters[2]),
          
          nn.ZeroPad2d((1,2,1,2)),
          nn.Conv2d(num_filters[2], num_filters[3], 5, 2, 0),
        )
    
    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters):
        super(synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
          nn.ZeroPad2d((1,0,1,0)),
          nn.ConvTranspose2d(in_dim, num_filters[0], 5, 2, 3, output_padding=1),
          GDN(num_filters[0], inverse=True),
          nn.ZeroPad2d((1,0,1,0)),
          nn.ConvTranspose2d(num_filters[0], num_filters[1], 5, 2, 3, output_padding=1),
          GDN(num_filters[1], inverse=True),
          nn.ZeroPad2d((1,0,1,0)),
          nn.ConvTranspose2d(num_filters[1], num_filters[2], 5, 2, 3, output_padding=1),
          GDN(num_filters[2], inverse=True)
        )
      
    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class Space2Depth(nn.Module):
    def __init__(self, r):
        super(Space2Depth, self).__init__()
        self.r = r
    
    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c * (r**2)
        out_h = h//2
        out_w = w//2
        x_view = x.view(b, c, out_h, r, out_w, r)
        x_prime = x_view.permute(0,3,5,1,2,4).contiguous().view(b, out_c, out_h, out_w)
        return x_prime

class Depth2Space(nn.Module):
    def __init__(self, r):
        super(Depth2Space, self).__init__()
        self.r = r
    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c // (r**2)
        out_h = h * 2
        out_w = w * 2
        x_view = x.view(b, r, r, out_c, h, w)
        x_prime = x_view.permute(0,3,4,1,5,2).contiguous().view(b, out_c, out_h, out_w)
        return x_prime

class h_analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
        super(h_analysisTransformModel, self).__init__()
        self.transform = nn.Sequential(
          nn.Conv2d(in_dim, num_filters[0], 3, strides_list[0], 1),
          Space2Depth(2),
          nn.Conv2d(num_filters[0]*4, num_filters[1], 1, strides_list[1], 0),
          nn.ReLU(),
          nn.Conv2d(num_filters[1], num_filters[1], 1, 1, 0),
          nn.ReLU(),
          nn.Conv2d(num_filters[1], num_filters[2], 1, 1, 0)
        )
      
    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class h_synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
        super(h_synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
          nn.ConvTranspose2d(in_dim, num_filters[0], 1, strides_list[2], 0),
          nn.ConvTranspose2d(num_filters[0], num_filters[1], 1, strides_list[1], 0),
          nn.ReLU(),
          nn.ConvTranspose2d(num_filters[1], num_filters[1], 1, strides_list[1], 0),
          nn.ReLU(),
          Depth2Space(2),
          nn.ZeroPad2d((0,0,0,0)),
          nn.ConvTranspose2d(num_filters[1]//4, num_filters[2], 3, strides_list[0], 1)
        )
  
    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class NeighborSample(nn.Module):
  def __init__(self):
    super(NeighborSample, self).__init__()
    self.unfolder = nn.Unfold(5, padding=2)
  
  def forward(self, inputs):
    b, c, h, w = inputs.size()
    t = self.unfolder(inputs) # (b, c*5*5, h*w)
    t = t.permute((0,2,1)).reshape(b*h*w, c, 5, 5)
    return t
  
class GaussianModel(nn.Module):
  def __init__(self):
    super(GaussianModel, self).__init__()
    
    self.m_normal_dist = torch.distributions.normal.Normal(0., 1.)
    # self.register_buffer('m_normal_dist', flt_tensor)

  def _cumulative(self, inputs, stds, mu):
    half = 0.5
    eps = 1e-6
    upper = (inputs - mu + half) / (stds)
    lower = (inputs - mu - half) / (stds)
    cdf_upper = self.m_normal_dist.cdf(upper)
    cdf_lower = self.m_normal_dist.cdf(lower)
    res = cdf_upper - cdf_lower
    return res
  
  def forward(self, inputs, hyper_sigma, hyper_mu):
    likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
    likelihood_bound = 1e-8
    likelihood = torch.clamp(likelihood, min=likelihood_bound)
    return likelihood
    
class PredictionModel(nn.Module):
  def __init__(self, in_dim, dim=192, trainable=True, outdim=None):
    super(PredictionModel, self).__init__()
    if outdim is None:
      outdim = dim
    self.transform = nn.Sequential(
      nn.ZeroPad2d((1,1,1,1)),
      nn.Conv2d(in_dim, dim, 3, 1, 0),
      nn.LeakyReLU(0.2),
      nn.ZeroPad2d((1,2,1,2)),
      nn.Conv2d(dim, dim, 3, 2, 0),
      nn.LeakyReLU(0.2),
      nn.ZeroPad2d((1,1,1,1)),
      nn.Conv2d(dim, dim, 3, 1, 0),
      nn.LeakyReLU(0.2)
    )
    self.fc = nn.Linear(dim*3*3, outdim)
    self.flatten = nn.Flatten()
    
  def forward(self, input_shape, h_tilde, h_sampler):
    b, c, h, w = input_shape
    h_sampled = h_sampler(h_tilde)
    h_sampled = self.transform(h_sampled)
    h_sampled = self.flatten(h_sampled)
    h_sampled = self.fc(h_sampled)
    hyper_mu = h_sampled[:, :c]
    hyper_mu = hyper_mu.view(b, h, w, c).permute(0, 3, 1, 2)
    hyper_sigma = h_sampled[:, c:]
    hyper_sigma = torch.exp(hyper_sigma)
    hyper_sigma = hyper_sigma.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

    return hyper_mu, hyper_sigma

class BypassRound(Function):
  @staticmethod
  def forward(ctx, inputs):
    return torch.round(inputs)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output

class SideInfoReconModel(nn.Module):
  def __init__(self, input_dim, num_filters=192):
    super(SideInfoReconModel, self).__init__()
    self.layer_1 = nn.Sequential(
      nn.ZeroPad2d((1,0,1,0)),
      nn.ConvTranspose2d(input_dim, num_filters, kernel_size=5, stride=2, padding=3, output_padding=1)
      )
    self.layer_1a = nn.Sequential(
      nn.ZeroPad2d((1,0,1,0)),
      nn.ConvTranspose2d(num_filters, num_filters, 5, 2, 3, output_padding=1),
      nn.LeakyReLU(0.2)
    )
    self.layer_1b = nn.Sequential(
      nn.ZeroPad2d((1,0,1,0)),
      nn.ConvTranspose2d(num_filters, num_filters, 5, 2, 3, output_padding=1),
      nn.LeakyReLU(0.2)
    )
    self.layer_3_1 = nn.Sequential(
      nn.Conv2d(num_filters*2, num_filters, 3, 1, 1),
      nn.LeakyReLU(0.2)
    )
    self.layer_3_2 = nn.Sequential(
      nn.Conv2d(num_filters, num_filters, 3, 1, 1),
      nn.LeakyReLU(0.2)
    )
    self.layer_3_3 = nn.Sequential(
      nn.Conv2d(num_filters, num_filters*2, 3, 1, 1),
      nn.LeakyReLU(0.2)
    )
    self.layer_4 = nn.Sequential(
      nn.ZeroPad2d((1,0,1,0)),
      nn.ConvTranspose2d(num_filters*2, num_filters//3, 5, 2, 3, output_padding=1)
    )
    self.layer_5 = nn.Sequential(
      nn.Conv2d(num_filters//3, num_filters//12, 3, 1, 1),
      nn.LeakyReLU(0.2)
    )
    self.layer_6 = nn.Conv2d(num_filters//12, 3, 1, 1, 0)
    self.d2s = Depth2Space(2)

  def forward(self, pf, h2, h1):
    h1prime = self.d2s(h1)
    h = torch.cat([h2, h1prime], 1)
    h = self.layer_1(h)
    h = self.layer_1a(h)
    h = self.layer_1b(h)

    hfeat_0 = torch.cat([pf, h], 1)
    hfeat = self.layer_3_1(hfeat_0)
    hfeat = self.layer_3_2(hfeat)
    hfeat = self.layer_3_3(hfeat)
    hfeat = hfeat_0 + hfeat

    x = self.layer_4(hfeat)
    x = self.layer_5(x)
    x = self.layer_6(x)
    return x
          
bypass_round = BypassRound.apply

# This is a trick to align encoding and decoding computation platforms
def washed(d):
    global device
    nd = d.detach().cpu().numpy()
    td = torch.tensor(nd).to(device)
    return td

class NetLow(nn.Module):
  def __init__(self):
    super(NetLow, self).__init__()

    self.a_model = analysisTransformModel(3, [192, 192, 192, 192])
    self.s_model = synthesisTransformModel(192, [192, 192, 192, 3])
    
    self.ha_model_1 = h_analysisTransformModel(64*4, [64*4, 32*4, 32*4], [1, 1, 1])
    self.hs_model_1 = h_synthesisTransformModel(32*4, [64*4, 64*4, 64*4], [1, 1, 1])

    self.ha_model_2 = h_analysisTransformModel(192, [384, 192*4, 64*4], [1, 1, 1])
    self.hs_model_2 = h_synthesisTransformModel(64*4, [192*4, 192*4, 192], [1, 1, 1])

    self.entropy_bottleneck_z1 = GaussianModel()
    self.entropy_bottleneck_z2 = GaussianModel()
    self.entropy_bottleneck_z3 = GaussianModel()

    self.h1_sigma = nn.Parameter(torch.ones((1,32*4,1,1), dtype=torch.float32, requires_grad=False))

    self.register_parameter('get_h1_sigma', self.h1_sigma)

    self.prediction_model_2 = PredictionModel(in_dim=64*4, dim=64*4, outdim=64*4*2)

    self.prediction_model_3 = PredictionModel(in_dim=192, dim=192, outdim=192*2)
    
    self.sampler_2 = NeighborSample()
    self.sampler_3 = NeighborSample()

    self.side_recon_model = SideInfoReconModel(192+64)

  def forward(self, inputs, mode='train'):
    pass
  
  def encode(self, inputs):
    b, c, h, w = inputs.shape
    tb, tc, th, tw = inputs.shape
    
    z3 = self.a_model(inputs)
    z3_rounded = bypass_round(z3)

    z2 = self.ha_model_2(z3_rounded)
    z2_rounded = bypass_round(z2)
    
    z1 = self.ha_model_1(z2_rounded)
    z1_rounded = bypass_round(z1)

    z1_sigma = torch.abs(self.get_h1_sigma)
    z1_mu = torch.zeros_like(z1_sigma)

    h1 = self.hs_model_1(washed(z1_rounded))
    h2 = self.hs_model_2(washed(z2_rounded))
    h3 = self.s_model(washed(z3_rounded))

    z1_likelihoods = self.entropy_bottleneck_z1(z1_rounded, z1_sigma, z1_mu)

    z2_mu, z2_sigma = self.prediction_model_2((tb, 64*4,th//2//16,tw//2//16), h1, self.sampler_2)

    z2_likelihoods = self.entropy_bottleneck_z2(z2_rounded, z2_sigma, z2_mu)
    
    z3_mu, z3_sigma = self.prediction_model_3((tb, 192,th//16,tw//16), h2, self.sampler_3)

    z3_likelihoods = self.entropy_bottleneck_z3(z3_rounded, z3_sigma, z3_mu)

    pf = self.s_model(z3_rounded)
    x_tilde = self.side_recon_model(pf, h2, h1)
    
    num_pixels = inputs.size()[0] * h * w
    
    test_num_pixels = inputs.size()[0] * h * w

    eval_bpp = torch.sum(torch.log(z3_likelihoods), [0,1,2,3]) / (-np.log(2) * test_num_pixels) + torch.sum(torch.log(z2_likelihoods), [0,1,2,3]) / (-np.log(2) * test_num_pixels) + torch.sum(torch.log(z1_likelihoods), [0,1,2,3]) / (-np.log(2) * test_num_pixels)

    gt = torch.round((inputs + 1) * 127.5)
    x_hat = torch.clamp((x_tilde + 1) * 127.5, 0, 255)
    x_hat = torch.round(x_hat).float()
    v_mse = torch.mean((x_hat - gt) ** 2, [1,2,3])
    v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)

    ret = {}
    ret['z1_mu'] = z1_mu.detach().cpu().numpy()
    ret['z1_sigma'] = z1_sigma.detach().cpu().numpy()
    ret['z2_mu'] = z2_mu.detach().cpu().numpy()
    ret['z2_sigma'] = z2_sigma.detach().cpu().numpy()
    ret['z3_mu'] = z3_mu.detach().cpu().numpy()
    ret['z3_sigma'] = z3_sigma.detach().cpu().numpy()
    ret['z1_rounded'] = z1_rounded.detach().cpu().numpy()
    ret['z2_rounded'] = z2_rounded.detach().cpu().numpy()
    ret['z3_rounded'] = z3_rounded.detach().cpu().numpy()
    ret['v_psnr'] = v_psnr.detach().cpu().numpy()
    ret['eval_bpp'] = eval_bpp.detach().cpu().numpy()
    return ret

  def decode(self, inputs, stage):
    if stage == 0:
      z1_sigma = torch.abs(self.get_h1_sigma)
      z1_mu = torch.zeros_like(z1_sigma)
      
      ret = {}
      ret['z1_sigma'] = z1_sigma.detach().cpu().numpy()
      ret['z1_mu'] = z1_mu.detach().cpu().numpy()
      return ret
    
    elif stage == 1:
      z1_rounded = inputs['z1_rounded']
      h1 = self.hs_model_1(z1_rounded)
      self.h1 = h1
      z2_mu, z2_sigma = self.prediction_model_2((h1.shape[0],64*4,h1.shape[2],h1.shape[3]), h1, self.sampler_2)
      ret = {}
      ret['z2_sigma'] = z2_sigma.detach().cpu().numpy()
      ret['z2_mu'] = z2_mu.detach().cpu().numpy()

      return ret
    
    elif stage == 2:
      z2_rounded = inputs['z2_rounded']
      h2 = self.hs_model_2(z2_rounded)
      self.h2 = h2
      z3_mu, z3_sigma = self.prediction_model_3((h2.shape[0],192,h2.shape[2],h2.shape[3]), h2, self.sampler_3)
      ret = {}
      ret['z3_sigma'] = z3_sigma.detach().cpu().numpy()
      ret['z3_mu'] = z3_mu.detach().cpu().numpy()
      return ret
    
    elif stage == 3:
      z3_rounded = inputs['z3_rounded']
      pf = self.s_model(z3_rounded)
      x_tilde = self.side_recon_model(pf, self.h2, self.h1)
      x_tilde = torch.round(torch.clamp((x_tilde + 1) * 127.5, 0, 255))
      return x_tilde.detach().cpu().numpy()

class NetHigh(nn.Module):
  def __init__(self):
    super(NetHigh, self).__init__()
    self.a_model = analysisTransformModel(3, [384, 384, 384, 384])
    self.s_model = synthesisTransformModel(384, [384, 384, 384, 3])
    
    self.ha_model_1 = h_analysisTransformModel(64*4, [64*4*2, 32*4*2, 32*4], [1, 1, 1])
    self.hs_model_1 = h_synthesisTransformModel(32*4, [64*4*2, 64*4*2, 64*4], [1, 1, 1])

    self.ha_model_2 = h_analysisTransformModel(384, [384*2, 192*4*2, 64*4], [1, 1, 1])
    self.hs_model_2 = h_synthesisTransformModel(64*4, [192*4*2, 192*4*2, 384], [1, 1, 1])

    self.entropy_bottleneck_z1 = GaussianModel()
    self.entropy_bottleneck_z2 = GaussianModel()
    self.entropy_bottleneck_z3 = GaussianModel()

    self.h1_sigma = nn.Parameter(torch.ones((1,32*4,1,1), dtype=torch.float32, requires_grad=False))

    self.register_parameter('get_h1_sigma', self.h1_sigma)

    self.prediction_model_2 = PredictionModel(in_dim=64*4, dim=64*4, outdim=64*4*2)

    self.prediction_model_3 = PredictionModel(in_dim=384, dim=384, outdim=384*2)
    
    self.sampler_2 = NeighborSample()
    self.sampler_3 = NeighborSample()

    self.side_recon_model = SideInfoReconModel(384+64, num_filters=384)

  def forward(self, inputs, mode='train'):
    pass
  
  def encode(self, inputs):
    b, c, h, w = inputs.shape
    tb, tc, th, tw = inputs.shape
    
    z3 = self.a_model(inputs)
    z3_rounded = bypass_round(z3)

    z2 = self.ha_model_2(z3_rounded)
    z2_rounded = bypass_round(z2)
    
    z1 = self.ha_model_1(z2_rounded)
    z1_rounded = bypass_round(z1)

    z1_sigma = torch.abs(self.get_h1_sigma)
    z1_mu = torch.zeros_like(z1_sigma)

    h1 = self.hs_model_1(washed(z1_rounded))
    h2 = self.hs_model_2(washed(z2_rounded))
    h3 = self.s_model(washed(z3_rounded))

    z1_likelihoods = self.entropy_bottleneck_z1(z1_rounded, z1_sigma, z1_mu)

    z2_mu, z2_sigma = self.prediction_model_2((tb,64*4,th//2//16,tw//2//16), h1, self.sampler_2)

    z2_likelihoods = self.entropy_bottleneck_z2(z2_rounded, z2_sigma, z2_mu)
    
    z3_mu, z3_sigma = self.prediction_model_3((tb,384,th//16,tw//16), h2, self.sampler_3)

    z3_likelihoods = self.entropy_bottleneck_z3(z3_rounded, z3_sigma, z3_mu)

    pf = self.s_model(z3_rounded)
    x_tilde = self.side_recon_model(pf, h2, h1)
    
    num_pixels = inputs.size()[0] * h * w
    
    test_num_pixels = inputs.size()[0] * h * w

    eval_bpp = torch.sum(torch.log(z3_likelihoods), [0,1,2,3]) / (-np.log(2) * test_num_pixels) + torch.sum(torch.log(z2_likelihoods), [0,1,2,3]) / (-np.log(2) * test_num_pixels) + torch.sum(torch.log(z1_likelihoods), [0,1,2,3]) / (-np.log(2) * test_num_pixels)

    gt = torch.round((inputs + 1) * 127.5)
    x_hat = torch.clamp((x_tilde + 1) * 127.5, 0, 255)
    x_hat = torch.round(x_hat).float()
    v_mse = torch.mean((x_hat - gt) ** 2, [1,2,3])
    v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)

    ret = {}
    ret['z1_mu'] = z1_mu.detach().cpu().numpy()
    ret['z1_sigma'] = z1_sigma.detach().cpu().numpy()
    ret['z2_mu'] = z2_mu.detach().cpu().numpy()
    ret['z2_sigma'] = z2_sigma.detach().cpu().numpy()
    ret['z3_mu'] = z3_mu.detach().cpu().numpy()
    ret['z3_sigma'] = z3_sigma.detach().cpu().numpy()
    ret['z1_rounded'] = z1_rounded.detach().cpu().numpy()
    ret['z2_rounded'] = z2_rounded.detach().cpu().numpy()
    ret['z3_rounded'] = z3_rounded.detach().cpu().numpy()
    ret['v_psnr'] = v_psnr.detach().cpu().numpy()
    ret['eval_bpp'] = eval_bpp.detach().cpu().numpy()
    return ret

  def decode(self, inputs, stage):
    if stage == 0:
      z1_sigma = torch.abs(self.get_h1_sigma)
      z1_mu = torch.zeros_like(z1_sigma)
      
      ret = {}
      ret['z1_sigma'] = z1_sigma.detach().cpu().numpy()
      ret['z1_mu'] = z1_mu.detach().cpu().numpy()
      return ret
    
    elif stage == 1:
      z1_rounded = inputs['z1_rounded']
      h1 = self.hs_model_1(z1_rounded)
      self.h1 = h1
      z2_mu, z2_sigma = self.prediction_model_2((h1.shape[0],64*4,h1.shape[2],h1.shape[3]), h1, self.sampler_2)
      ret = {}
      ret['z2_sigma'] = z2_sigma.detach().cpu().numpy()
      ret['z2_mu'] = z2_mu.detach().cpu().numpy()

      return ret
    
    elif stage == 2:
      z2_rounded = inputs['z2_rounded']
      h2 = self.hs_model_2(z2_rounded)
      self.h2 = h2
      z3_mu, z3_sigma = self.prediction_model_3((h2.shape[0],384,h2.shape[2],h2.shape[3]), h2, self.sampler_3)
      ret = {}
      ret['z3_sigma'] = z3_sigma.detach().cpu().numpy()
      ret['z3_mu'] = z3_mu.detach().cpu().numpy()
      return ret
    
    elif stage == 3:
      z3_rounded = inputs['z3_rounded']
      pf = self.s_model(z3_rounded)
      x_tilde = self.side_recon_model(pf, self.h2, self.h1)
      x_tilde = torch.round(torch.clamp((x_tilde + 1) * 127.5, 0, 255))
      return x_tilde.detach().cpu().numpy()

def get_partition(h, w):
    hs = []
    eh = h

    hs += [768,] * (eh // 768)
    eh = eh % 768
    if eh != 0:
        hs += [eh,]
    ew = w
    ws = [768,] * (ew // 768)
    ew = ew % 768
    if ew != 0:
        ws += [ew,]

    print(hs, ws)
    return hs, ws

EXE_ARITH = './module_arithmeticcoding'
def compress_low(args):
  """Compresses an image."""
  global device
  mode = 'low'
  if args.qp > 3:
    mode = 'high'
  device = torch.device(args.device)
  from PIL import Image
  # Load input image and add batch dimension.
  f = Image.open(args.input)
  fshape = [f.size[1], f.size[0], 3]
  x = np.array(f).reshape([1,] + fshape)

  compressed_file_path = args.output
  fileobj = open(compressed_file_path, mode='wb')

  qp = args.qp
  model_type = args.model_type
  print(f'model_type: {model_type}, qp: {qp}')
 

  buf = qp << 1
  buf = buf + model_type
  arr = np.array([0], dtype=np.uint8)
  arr[0] =  buf
  arr.tofile(fileobj)

  h, w = (fshape[0]//64)*64, (fshape[1]//64)*64
  if h < fshape[0]:
    h += 64
  if w < fshape[1]:
    w += 64

  pad_up = (h - fshape[0]) // 2
  pad_down = (h - fshape[0]) - pad_up
  pad_left = (w - fshape[1]) // 2
  pad_right = (w - fshape[1]) - pad_left

  x = np.pad(x, [[0, 0], [pad_up, pad_down], [pad_left, pad_right], [0, 0]])
  torch_x = (torch.tensor(x).permute(0,3,1,2).float() / 127.5) - 1
  
  if mode == 'low':
    net = NetLow().eval()
  else:
    net = NetHigh().eval()
  net = net.to(device)

  sd = net.state_dict()
  td = load_tf_weights(sd, f'models/model{model_type}_qp{qp}.pk')
  net.load_state_dict(td)

  arr = np.array([fshape[0], fshape[1]], dtype=np.uint16)
  arr.tofile(fileobj)
  fileobj.close()

  hs, ws = get_partition(h, w)
  blocks = []

  ph = 0
  for th in hs:
    pw = 0
    for tw in ws:
      blocks.append(torch_x[:,:,ph:ph+th,pw:pw+tw])
      pw += tw
    ph += th
  
  cnt = 0
  for block in tqdm.tqdm(blocks):
    with torch.no_grad():
      block = block.to(device)
      ret = net.encode(block)
      bsf = open(compressed_file_path, "ab+")
      v_z1_sigma = ret['z1_sigma']
      v_z1_rounded = ret['z1_rounded']

      sigma_val = np.reshape(v_z1_sigma, (1,-1,1,1))
      tshape = v_z1_rounded.shape
      sigma_val = np.tile(sigma_val, (tshape[0], 1, tshape[2], tshape[3]))
      mu_val = np.zeros_like(sigma_val)
      mu_val[...] = 511
      flat_sigma = sigma_val.reshape((-1))
      flat_mu = mu_val.reshape((-1))
      flat_coeff = (v_z1_rounded + 511).astype(np.int16).reshape((-1))
      length = np.array([flat_sigma.shape[0]], dtype=np.int64)
      s = length.tobytes() + flat_coeff.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes()   
      r = sp.run([EXE_ARITH,'e',f'{mask_a}',f'{mask_b}'], input=s, stdout=sp.PIPE)
      bslen = np.array([len(r.stdout)], dtype=np.int64)
      bsf.write(bslen.tobytes())
      masks = np.array([mask_a,mask_b], dtype=np.uint8)
      bsf.write(masks.tobytes())
      bsf.write(r.stdout)
      # print(length, bslen)

      v_z2_sigma = ret['z2_sigma']
      v_z2_mu = ret['z2_mu']
      v_z2_rounded = ret['z2_rounded']
        
      sigma_val = v_z2_sigma
      tshape = v_z2_rounded.shape
      mu_val = v_z2_mu + 511
      flat_sigma = sigma_val.reshape((-1))
      flat_mu = mu_val.reshape((-1))
      flat_coeff = (v_z2_rounded + 511).astype(np.int16).reshape((-1))
      length = np.array([flat_sigma.shape[0]], dtype=np.int64)
      s = length.tobytes() + flat_coeff.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes()   
      r = sp.run([EXE_ARITH,'e',f'{mask_a}',f'{mask_b}'], input=s, stdout=sp.PIPE)
      bslen = np.array([len(r.stdout)], dtype=np.int64)
      bsf.write(bslen.tobytes())
      masks = np.array([mask_a,mask_b], dtype=np.uint8)
      bsf.write(masks.tobytes())
      bsf.write(r.stdout)
      # print(length, bslen)

      v_z3_sigma = ret['z3_sigma']
      v_z3_mu = ret['z3_mu']
      v_z3_rounded = ret['z3_rounded']

      sigma_val = v_z3_sigma
      tshape = v_z3_rounded.shape
      mu_val = v_z3_mu + 511
      flat_sigma = sigma_val.reshape((-1))
      flat_mu = mu_val.reshape((-1))
      flat_coeff = (v_z3_rounded + 511).astype(np.int16).reshape((-1))
      length = np.array([flat_sigma.shape[0]], dtype=np.int64)
      s = length.tobytes() + flat_coeff.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes()   
      r = sp.run([EXE_ARITH,'e',f'{mask_a}',f'{mask_b}'], input=s, stdout=sp.PIPE)
      bslen = np.array([len(r.stdout)], dtype=np.int64)
      bsf.write(bslen.tobytes())
      masks = np.array([mask_a,mask_b], dtype=np.uint8)
      bsf.write(masks.tobytes())
      bsf.write(r.stdout)
      # print(length, bslen)
      
      bsf.close()
      cnt += 1

def decompress_low(args):
  from PIL import Image
  global device

  device = torch.device(args.device)
  compressed_file = args.input
  fileobj = open(compressed_file, mode='rb')
  buf = fileobj.read(1)
  arr = np.frombuffer(buf, dtype=np.uint8)
  model_type = arr[0] % 2
  qp = arr[0] >> 1
  mode = 'low'
  if qp > 3:
    mode = 'high'
  print(f'model_type: {model_type}, qp: {qp}')

  buf = fileobj.read(4)
  arr = np.frombuffer(buf, dtype=np.uint16)
  h = int(arr[0])
  w = int(arr[1])
  oh = h
  ow = w

  fshape = [h, w]

  h, w = (fshape[0]//64)*64, (fshape[1]//64)*64
  if h < fshape[0]:
    h += 64
  if w < fshape[1]:
    w += 64

  pad_up = (h - fshape[0]) // 2
  pad_down = (h - fshape[0]) - pad_up
  pad_left = (w - fshape[1]) // 2
  pad_right = (w - fshape[1]) - pad_left
  
  padded_w = int(math.ceil(w / 16) * 16)
  padded_h = int(math.ceil(h / 16) * 16)

  if mode == 'low':
    net = NetLow()
  else:
    net = NetHigh()
  net = net.to(device)
  
  sd = net.state_dict()
  td = load_tf_weights(sd, f'models/model{model_type}_qp{qp}.pk')
  net.load_state_dict(td)

  def decode_block(padded_h, padded_w):
    with torch.no_grad():
      z1_hat = np.zeros((1, 32*4, padded_h//64, padded_w//64), dtype=np.float32)
      z2_hat = np.zeros((1, 64*4, padded_h//32, padded_w//32), dtype=np.float32)
      if mode == 'low':
        z3_hat = np.zeros((1, 192, padded_h//16, padded_w//16) , dtype=np.float32)
      else:
        z3_hat = np.zeros((1, 384, padded_h//16, padded_w//16) , dtype=np.float32)

      ret = net.decode(None, 0)
      sigma_z = ret['z1_sigma']

      bslenbs = fileobj.read(8)
      bslen = np.frombuffer(bslenbs, dtype=np.int64)
      maskbs = fileobj.read(2)
      masks = np.frombuffer(maskbs, dtype=np.uint8)
      bs = fileobj.read(bslen[0])
      length = np.array(z1_hat.reshape((-1)).shape[0], dtype=np.int64)
      sigma_val = np.reshape(sigma_z, (1,-1,1,1))
      tshape = z1_hat.shape
      sigma_val = np.tile(sigma_val, (tshape[0], 1, tshape[2], tshape[3]))
      mu_val = np.zeros_like(sigma_val)
      mu_val[...] = 511
      flat_sigma = sigma_val.reshape((-1))
      flat_mu = mu_val.reshape((-1))
      s = length.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes() + bs
      r = sp.run(['./module_arithmeticcoding','d',f'{masks[0]}',f'{masks[1]}'], input=s, stdout=sp.PIPE)
      coeffs = np.frombuffer(r.stdout, dtype=np.int16)
      z1_hat[...] = coeffs.reshape(z1_hat.shape).astype(np.float32) - 511
      

      s2_in = {'z1_rounded':torch.tensor(z1_hat).to(device)}
      
      ret = net.decode(s2_in, 1)
      v_z2_sigma = ret['z2_sigma']
      v_z2_mu = ret['z2_mu']

      bslenbs = fileobj.read(8)
      bslen = np.frombuffer(bslenbs, dtype=np.int64)
      maskbs = fileobj.read(2)
      masks = np.frombuffer(maskbs, dtype=np.uint8)
      bs = fileobj.read(bslen[0])
      length = np.array(z2_hat.reshape((-1)).shape[0], dtype=np.int64)
      sigma_val = v_z2_sigma
      tshape = z2_hat.shape
      mu_val = v_z2_mu + 511
      flat_sigma = sigma_val.reshape((-1))
      flat_mu = mu_val.reshape((-1))
      s = length.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes() + bs
      r = sp.run(['./module_arithmeticcoding','d',f'{masks[0]}',f'{masks[1]}'], input=s, stdout=sp.PIPE)
      coeffs = np.frombuffer(r.stdout, dtype=np.int16)
      z2_hat[...] = coeffs.reshape(z2_hat.shape).astype(np.float32) - 511

      s3_in = {'z2_rounded':torch.tensor(z2_hat).to(device)}
      ret = net.decode(s3_in, 2)
      v_z3_sigma = ret['z3_sigma']
      v_z3_mu = ret['z3_mu']

      bslenbs = fileobj.read(8)
      bslen = np.frombuffer(bslenbs, dtype=np.int64)
      maskbs = fileobj.read(2)
      masks = np.frombuffer(maskbs, dtype=np.uint8)
      bs = fileobj.read(bslen[0])
      length = np.array(z3_hat.reshape((-1)).shape[0], dtype=np.int64)
      sigma_val = v_z3_sigma
      tshape = z3_hat.shape
      mu_val = v_z3_mu + 511
      flat_sigma = sigma_val.reshape((-1))
      flat_mu = mu_val.reshape((-1))
      s = length.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes() + bs
      r = sp.run(['./module_arithmeticcoding','d',f'{masks[0]}',f'{masks[1]}'], input=s, stdout=sp.PIPE)
      coeffs = np.frombuffer(r.stdout, dtype=np.int16)
      z3_hat[...] = coeffs.reshape(z3_hat.shape).astype(np.float32) - 511

      s4_in = {'z3_rounded':torch.tensor(z3_hat).to(device)}
      x_tilde = net.decode(s4_in, 3)
      return x_tilde

  hs, ws = get_partition(padded_h, padded_w)
  blocks = []
  
  tar = np.zeros((1, 3, padded_h, padded_w), dtype=np.float32)
  ph = 0
  for th in hs:
    pw = 0
    for tw in ws:
      tar[:,:,ph:ph+th,pw:pw+tw] = decode_block(th, tw)
      pw += tw
    ph += th

  if qp < 3:
    ph = 0
    for th in hs[:-1]:
      ph += th
      blks = []
      for x in range(0, padded_w-8, 4):
        blk = tar[0,:,ph-4:ph+4,x:x+8]
        im = blk.astype(np.uint8).transpose((1,2,0))
        img = Image.fromarray(im)
        img = img.filter(ImageFilter.SMOOTH)
        imt = np.array(img).transpose((2,0,1)).astype(np.float32)
        blks.append(imt)
      cnt = 0
      for x in range(0, padded_w-8, 4):
        tar[0,:,ph-2:ph+2,x+2:x+6] = blks[cnt][:,2:-2,2:-2]
        cnt += 1
    
    pw = 0
    for tw in ws[:-1]:
      pw += tw
      blks = []
      for y in range(0, padded_h-8, 4):
        blk = tar[0,:,y:y+8,pw-4:pw+4]
        im = blk.astype(np.uint8).transpose((1,2,0))
        img = Image.fromarray(im)
        img = img.filter(ImageFilter.SMOOTH)
        imt = np.array(img).transpose((2,0,1)).astype(np.float32)
        blks.append(imt)
      cnt = 0
      for y in range(0, padded_h-8, 4):
        tar[0,:,y+2:y+6,pw-2:pw+2] = blks[cnt][:,2:-2,2:-2]
        cnt += 1

  def make_img(x_tilde, fn):
    im = x_tilde.astype(np.uint8).transpose((1,2,0))
    img = Image.fromarray(im)
    img.save(fn)

  if pad_down != 0 and pad_right != 0:
    make_img(tar[0, :, pad_up:-pad_down,pad_left:-pad_right], args.output)
  elif pad_down != 0:
    make_img(tar[0, :, pad_up:-pad_down,pad_left:], args.output)
  elif pad_right != 0:
    make_img(tar[0, :, pad_up:,pad_left:-pad_right], args.output)
  else:
    make_img(tar[0, :, pad_up:,pad_left:], args.output)
