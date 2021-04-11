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

from torch.autograd import Function
import time
import os

from gdn_v3 import GDN, IGDN

class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, train_glob, transform):
        super(DIV2KDataset, self).__init__()
        self.transform = transform
        self.images = list(sorted(glob.glob(train_glob)))

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

class Preprocess(object):
    def __init__(self):
        pass

    def __call__(self, PIL_img):
        img = np.asarray(PIL_img, dtype=np.float32)
        img /= 127.5
        img -= 1.0
        return img.transpose((2, 0, 1))

# Main analysis transform model with GDN
class analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, conv_trainable=True):
        super(analysisTransformModel, self).__init__()
        self.t0 = nn.Sequential(
        )
        self.transform = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_dim, num_filters[0], 5, 2, 0),
            GDN(num_filters[0]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[0], num_filters[1], 5, 2, 0),
            GDN(num_filters[1]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[1], num_filters[2], 5, 2, 0),
            GDN(num_filters[2]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[2], num_filters[3], 5, 2, 0),
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x

# Main synthesis transform model with IGDN
class synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, conv_trainable=True):
        super(synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(
                in_dim, num_filters[0], 5, 2, 3, output_padding=1),
            IGDN(num_filters[0]),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(
                num_filters[0], num_filters[1], 5, 2, 3, output_padding=1),
            IGDN(num_filters[1]),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(
                num_filters[1], num_filters[2], 5, 2, 3, output_padding=1),
            IGDN(num_filters[2])
        )
        # Auxiliary convolution layer: the final layer of the synthesis model.
        # Used only in the initial training stages, when the information 
        # aggregation reconstruction module is not yet enabled.
        self.aux_conv = nn.Sequential(
          nn.ZeroPad2d((1,0,1,0)),
          nn.ConvTranspose2d(num_filters[2], num_filters[3], 5, 2, 3, output_padding=1)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        y = self.aux_conv(x)
        return x, y

# Space-to-depth & depth-to-space module
# same to TensorFlow implementations
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
        x_prime = x_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(
            b, out_c, out_h, out_w)
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
        x_prime = x_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(
            b, out_c, out_h, out_w)
        return x_prime

# Hyper analysis transform (w/o GDN)
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

# Hyper synthesis transform (w/o GDN)
class h_synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
        super(h_synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(in_dim, num_filters[0], 1, strides_list[2], 0),
            nn.ConvTranspose2d(
                num_filters[0], num_filters[1], 1, strides_list[1], 0),
            nn.ReLU(),
            nn.ConvTranspose2d(
                num_filters[1], num_filters[1], 1, strides_list[1], 0),
            nn.ReLU(),
            Depth2Space(2),
            nn.ZeroPad2d((0, 0, 0, 0)),
            nn.ConvTranspose2d(
                num_filters[1]//4, num_filters[2], 3, strides_list[0], 1)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x

# Sliding window module
class NeighborSample(nn.Module):
    def __init__(self, in_shape):
        super(NeighborSample, self).__init__()
        self.unfolder = nn.Unfold(5, padding=2)
  
    def forward(self, inputs):
        b, c, h, w = inputs.size()
        t = self.unfolder(inputs) # (b, c*5*5, h*w)
        t = t.permute((0,2,1)).reshape(b*h*w, c, 5, 5)
        return t

# Gaussian likelihood calculation module
class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        self.m_normal_dist = torch.distributions.normal.Normal(0., 1.)

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

# Prediction module to generate mean and scale for entropy coding
class PredictionModel(nn.Module):
    def __init__(self, in_dim, dim=192, trainable=True, outdim=None):
        super(PredictionModel, self).__init__()
        if outdim is None:
            outdim = dim
        self.transform = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_dim, dim, 3, 1, 0),
            nn.LeakyReLU(0.2),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(dim, dim, 3, 2, 0),
            nn.LeakyReLU(0.2),
            nn.ZeroPad2d((1, 1, 1, 1)),
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

# differentiable rounding function
class BypassRound(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# Information-Aggregation Reconstruction network
class SideInfoReconModel(nn.Module):
    def __init__(self, input_dim, num_filters=192):
        super(SideInfoReconModel, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(input_dim, num_filters, kernel_size=5,
                               stride=2, padding=3, output_padding=1)
        )
        self.layer_1a = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters, num_filters,
                               5, 2, 3, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.layer_1b = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters, num_filters,
                               5, 2, 3, output_padding=1),
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
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters*2, num_filters //
                               3, 5, 2, 3, output_padding=1)
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

# Projection head for constructing helping loss.
# It is used to apply a constraint to the decoded hyperprior.
# It helps hyperpriors preserve vital information during multi-layer training.
class ProjHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjHead, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        )
    def forward(self, inputs):
        return self.transform(inputs)

# Utility to align encoding and decoding devices
def washed(d, device=torch.device('cuda')):
    nd = d.detach().cpu().numpy()
    td = torch.tensor(nd).to(device)
    return td

# Main network
# The current hyper parameters are for higher-bit-rate compression (2x)
# Stage 1: train the main encoder & decoder, fine hyperprior
# Stage 2: train the whole network w/o info-agg sub-network
# Stage 3: disable the final layer of the synthesis transform and enable info-agg net
# Stage 4: End-to-end train the whole network w/o the helping (auxillary) loss
class Net(nn.Module):
    def __init__(self, train_size=(1,256,256,3), test_size=(1,256,256,3)):
        super(Net, self).__init__()
        self.train_size = train_size
        self.test_size = test_size
        self.a_model = analysisTransformModel(
            3, [384, 384, 384, 384])
        self.s_model = synthesisTransformModel(
            384, [384, 384, 384, 3])

        self.ha_model_1 = h_analysisTransformModel(
            64*4, [64*4*2, 32*4*2, 32*4], [1, 1, 1])
        self.hs_model_1 = h_synthesisTransformModel(
            32*4, [64*4*2, 64*4*2, 64*4], [1, 1, 1])

        self.ha_model_2 = h_analysisTransformModel(
            384, [384*2, 192*4*2, 64*4], [1, 1, 1])
        self.hs_model_2 = h_synthesisTransformModel(
            64*4, [192*4*2, 192*4*2, 384], [1, 1, 1])

        self.entropy_bottleneck_z1 = GaussianModel()
        self.entropy_bottleneck_z2 = GaussianModel()
        self.entropy_bottleneck_z3 = GaussianModel()
        b, h, w, c = train_size
        tb, th, tw, tc = test_size

        self.h1_sigma = torch.nn.Parameter(torch.ones(
            (1, 32*4, 1, 1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('get_h1_sigma', self.h1_sigma)

        self.v_z2_sigma = torch.nn.Parameter(torch.ones(
            (1, 64*4, 1, 1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('z2_sigma', self.v_z2_sigma)

        self.prediction_model_2 = PredictionModel(
            in_dim=64*4, dim=64*4, outdim=64*4*2)

        self.prediction_model_3 = PredictionModel(
            in_dim=384, dim=384, outdim=384*2)

        self.sampler_2 = NeighborSample((b, 64*4, h//2//8, w//2//8))
        self.sampler_3 = NeighborSample((b, 384, h//8, w//8))

        self.side_recon_model = SideInfoReconModel(384+64, num_filters=384)

        self.proj_head_z3 = ProjHead(384, 384)
        self.proj_head_z2 = ProjHead(64*4, 64*4)
    
    def stage1_params(self):
        params = []
        for v in self.a_model.parameters():
            params.append(v)
        for v in self.s_model.parameters():
            params.append(v)

        for v in self.ha_model_2.parameters():
            params.append(v)
        for v in self.hs_model_2.parameters():
            params.append(v)
        for v in self.proj_head_z3.parameters():
            params.append(v)
        for v in self.prediction_model_3.parameters():
            params.append(v)
        params.append(self.z2_sigma)

        return params
    
    def stage2_params(self):
        params = []

        for v in self.a_model.parameters():
            params.append(v)
        for v in self.s_model.parameters():
            params.append(v)

        for v in self.ha_model_2.parameters():
            params.append(v)
        for v in self.hs_model_2.parameters():
            params.append(v)
        for v in self.proj_head_z3.parameters():
            params.append(v)
        for v in self.prediction_model_3.parameters():
            params.append(v)

        for v in self.ha_model_1.parameters():
            params.append(v)
        for v in self.hs_model_1.parameters():
            params.append(v)
        for v in self.proj_head_z2.parameters():
            params.append(v)
        for v in self.prediction_model_2.parameters():
            params.append(v)
        params.append(self.get_h1_sigma)

        return params

    # We adopt a multi-stage training procedure
    def forward(self, inputs, mode='train', stage=1):
        b, h, w, c = self.train_size
        tb, th, tw, tc = self.test_size

        z3 = self.a_model(inputs)
        noise = torch.rand_like(z3) - 0.5
        z3_noisy = z3 + noise
        z3_rounded = bypass_round(z3)

        z2 = self.ha_model_2(z3_rounded)
        noise = torch.rand_like(z2) - 0.5
        z2_noisy = z2 + noise
        z2_rounded = bypass_round(z2)

        if stage > 1: # h1 enabled after stage 2
            z1 = self.ha_model_1(z2_rounded)
            noise = torch.rand_like(z1) - 0.5
            z1_noisy = z1 + noise
            z1_rounded = bypass_round(z1)

            z1_sigma = torch.abs(self.get_h1_sigma)
            z1_mu = torch.zeros_like(z1_sigma)

            h1 = self.hs_model_1(z1_rounded)
            if stage < 4:
                proj_z2 = self.proj_head_z2(h1)
        
        h2 = self.hs_model_2(z2_rounded)
        if stage < 4: # helping loss enabled before stage 4
            proj_z3 = self.proj_head_z3(h2)

        if mode == 'train':
            if stage > 1: # when h1 enabled after stage 2
                z1_likelihoods = self.entropy_bottleneck_z1(
                    z1_noisy, z1_sigma, z1_mu)

                z2_mu, z2_sigma = self.prediction_model_2(
                    (b, 64*4, h//2//16, w//2//16), h1, self.sampler_2)
            else:
                z2_sigma = torch.abs(self.z2_sigma)
                z2_mu = torch.zeros_like(z2_sigma)

            z2_likelihoods = self.entropy_bottleneck_z2(
                z2_noisy, z2_sigma, z2_mu)

            z3_mu, z3_sigma = self.prediction_model_3(
                (b, 384, h//16, w//16), h2, self.sampler_3)

            z3_likelihoods = self.entropy_bottleneck_z3(
                z3_noisy, z3_sigma, z3_mu)
        else:
            if stage > 1: # when h1 enabled after stage 2
                z1_likelihoods = self.entropy_bottleneck_z1(
                    z1_rounded, z1_sigma, z1_mu)

                z2_mu, z2_sigma = self.prediction_model_2(
                    (tb, 64*4, th//2//16, tw//2//16), h1, self.sampler_2)
            else:
                z2_sigma = torch.abs(self.z2_sigma)
                z2_mu = torch.zeros_like(z2_sigma)

            z2_likelihoods = self.entropy_bottleneck_z2(
                z2_rounded, z2_sigma, z2_mu)

            z3_mu, z3_sigma = self.prediction_model_3(
                (tb, 384, th//16, tw//16), h2, self.sampler_3)

            z3_likelihoods = self.entropy_bottleneck_z3(
                z3_rounded, z3_sigma, z3_mu)

        if stage <= 2: # when side recon model not enabled
            pf, y = self.s_model(z3_rounded)
            x_tilde = y
        elif stage >= 3: # side-info recon model on
            pf, y = self.s_model(z3_rounded)
            x_tilde = self.side_recon_model(pf, h2, h1)

        num_pixels = inputs.size()[0] * h * w

        if mode == 'train':

            train_mse = torch.mean((inputs - x_tilde) ** 2, [0, 1, 2, 3])
            train_mse *= 255**2

            if stage == 1: # RDO on h2 and h3
                bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)
                        for l in [z2_likelihoods, z3_likelihoods]]
                train_bpp = bpp_list[0] + bpp_list[1]
                train_aux3 = torch.nn.MSELoss(reduction='mean')(z3.detach(), proj_z3)
                train_loss = args.lmbda * train_mse + train_bpp + train_aux3

            elif stage == 2: # Full RDO on h1, h2 and h3
                bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)
                        for l in [z1_likelihoods, z2_likelihoods, z3_likelihoods]]
                train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]
                train_aux3 = torch.nn.MSELoss(reduction='mean')(z3.detach(), proj_z3)
                train_aux2 = torch.nn.MSELoss(reduction='mean')(z2.detach(), proj_z2)
                train_loss = args.lmbda * train_mse + train_bpp + train_aux2 + train_aux3
            elif stage == 3: # with side recon model; full RDO
                bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)
                            for l in [z1_likelihoods, z2_likelihoods, z3_likelihoods]]
                train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]
                train_aux3 = torch.nn.MSELoss(reduction='mean')(z3.detach(), proj_z3)
                train_aux2 = torch.nn.MSELoss(reduction='mean')(z2.detach(), proj_z2)
                train_loss = args.lmbda * train_mse + train_bpp + train_aux2 + train_aux3
            else: # no aux loss
                bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)
                            for l in [z1_likelihoods, z2_likelihoods, z3_likelihoods]]
                train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]
                train_loss = args.lmbda * train_mse + train_bpp

            return train_loss, train_bpp, train_mse

        elif mode == 'test':
            test_num_pixels = inputs.size()[0] * 256 ** 2

            if stage == 1:
                eval_bpp = torch.sum(torch.log(z3_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels) + torch.sum(torch.log(z2_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels)
                
                bpp3 = torch.sum(torch.log(z3_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels)
                bpp2 = torch.sum(torch.log(z2_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels)
                bpp1 = torch.zeros_like(bpp2)
                
            else:
                eval_bpp = torch.sum(torch.log(z3_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels) + torch.sum(torch.log(z2_likelihoods), [
                    0, 1, 2, 3]) / (-np.log(2) * test_num_pixels) + torch.sum(torch.log(z1_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels)
                
                bpp3 = torch.sum(torch.log(z3_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels)
                bpp2 = torch.sum(torch.log(z2_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels)
                bpp1 = torch.sum(torch.log(z1_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels)

            # Bring both images back to 0..255 range.
            gt = torch.round((inputs + 1) * 127.5)
            x_hat = torch.clamp((x_tilde + 1) * 127.5, 0, 255)
            x_hat = torch.round(x_hat).float()
            v_mse = torch.mean((x_hat - gt) ** 2, [1, 2, 3])
            v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)
            return eval_bpp, v_psnr, x_hat, bpp1, bpp2, bpp3

def train():
    device = torch.device('cuda')
    train_data = DIV2KDataset(
        args.train_glob, transform=tv.transforms.Compose([
            tv.transforms.RandomCrop(256),
            Preprocess()
        ])
    )
    training_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsize, shuffle=True, num_workers=8
    )
    
    # In this version, parallel level should be manually set in the code.
    # The example if for training with 2 GPUs
    n_parallel = 2
    net = nn.DataParallel(Net((args.batchsize//n_parallel, 256, 256, 3),
              (8//n_parallel, 256, 256, 3)).to(device))

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    if args.load_weights != '':
        net.apply(weight_init)
        d = torch.load(args.load_weights)
        nd = {}
        for k in d.keys():
            if 'proj_head_z2' in k:
                continue
            nd[k] = d[k]
        net.load_state_dict(nd, strict=False)
    else:
        net.apply(weight_init)
    
    opt = optim.Adam(net.parameters(), lr=1e-4)
    opt_s1 = optim.Adam(net.module.stage1_params(), lr=1e-4)
    opt_s2 = optim.Adam(net.module.stage2_params(), lr=1e-4)

    sch = optim.lr_scheduler.MultiStepLR(opt, [4000-1300, 4500-1300, 4750-1300, 5000-1300], 0.5)

    # for checkpoint resume
    stage = 1
    st_epoch = 0
    ###################
    for epoch in range(st_epoch, 5500):
        start_time = time.time()
        list_train_loss = 0.
        list_train_bpp = 0.
        list_train_mse = 0.
        list_train_aux = 0.

        cnt = 0

        for i, data in enumerate(training_loader, 0):
            x = data.to(device)
            opt.zero_grad()
            train_loss, train_bpp, train_mse = net(x, 'train', stage=stage)

            train_loss = train_loss.mean()
            train_bpp = train_bpp.mean()
            train_mse = train_mse.mean()

            if np.isnan(train_loss.item()):
                raise Exception('NaN in loss')

            list_train_loss += train_loss.item()
            list_train_bpp += train_bpp.item()
            list_train_mse += train_mse.item()

            train_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10)
            if stage == 1:
                opt_s1.step()
            elif stage == 2:
                opt_s2.step()
            else:
                opt.step()
            cnt += 1

        timestamp = time.time()
        print('[Epoch %04d TRAIN %.1f seconds] Loss: %.4e bpp: %.4e mse: %.4e aux: %.4e' % (
            epoch,
            timestamp - start_time,
            list_train_loss / cnt,
            list_train_bpp / cnt,
            list_train_mse / cnt,
            list_train_aux / cnt
        )
        )

        if epoch % 100 == 0:
            print('[INFO] Saving')
            if not os.path.isdir(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            torch.save(net.state_dict(), './%s/%04d.ckpt' %
                       (args.checkpoint_dir, epoch))
            torch.save(opt.state_dict(), './%s/latest_opt.ckpt' % (args.checkpoint_dir))
            torch.save(opt_s1.state_dict(), './%s/latest_opts1.ckpt' % (args.checkpoint_dir))
            torch.save(opt_s2.state_dict(), './%s/latest_opts2.ckpt' % (args.checkpoint_dir))
        
        if epoch == 1000:
            stage = 2
        elif epoch == 1200:
            stage = 3
        elif epoch == 1300:
            stage = 4
        if stage >= 3:
            sch.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "command", choices=["train"],
        help="What to do?")
    parser.add_argument(
        "input", nargs="?",
        help="Input filename.")
    parser.add_argument(
        "output", nargs="?",
        help="Output filename.")
    parser.add_argument(
        "--num_filters", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")
    parser.add_argument(
        "--train_glob", default="images/*.png",
        help="Glob pattern identifying training data. This pattern must expand "
             "to a list of RGB images in PNG format.")
    parser.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training.")
    parser.add_argument(
        "--patchsize", type=int, default=256,
        help="Size of image patches for training.")
    parser.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    parser.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")

    parser.add_argument(
        "--load_weights", default="",
        help="Loaded weights")

    args = parser.parse_args()

    if args.command == "train":
        train()
