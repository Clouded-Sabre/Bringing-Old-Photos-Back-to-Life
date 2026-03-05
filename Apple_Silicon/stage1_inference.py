#!/usr/bin/env python3
"""
Stage 1 Inference Script for Apple Silicon (Mac M1/M2/M3/M4)
Bringing Old Photos Back to Life - Quality Restoration

Supports all arguments from Global/test.py for Stage 1 inference.

Usage:
    python stage1_inference.py --test_input /path/to/input --outputs_dir /path/to/output [options]

See --help for all available options.
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import cv2

# Import original models from the codebase
from Global.models.networks import GlobalGenerator_DCDCv2
from Global.models import networks
from Global.models.base_model import BaseModel


class ModelOptions:
    """Simple options class to mimic the original opt structure"""
    def __init__(self):
        self.ngf = 64
        self.n_downsample_global = 3
        self.mc = 64
        self.k_size = 4
        self.start_r = 1
        self.spatio_size = 32
        self.norm = 'instance'
        self.label_nc = 0
        self.output_nc = 3
        self.input_nc = 3
        self.feat_dim = -1
        self.mapping_net_dilation = 1
        self.gpu_ids = []
        self.checkpoints_dir = './checkpoints'
        self.name = ''
        self.use_vae_which_epoch = 'latest'
        self.which_epoch = 'latest'
        self.load_pretrainA = ''
        self.load_pretrainB = ''
        self.load_pretrain = ''
        self.NL_use_mask = False

# Keep detection model classes
class Downsample(nn.Module):
    def __init__(self, pad_type="reflect", filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        if self.filt_size == 1:
            a = np.array([1.0,])
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            return inp[:, :, :: self.stride, :: self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class UNetConvBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        for _ in range(conv_num):
            block.append(nn.ReflectionPad2d(padding=int(padding)))
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=0))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.LeakyReLU(0.2, True))
            in_size = out_size

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=0),
            )

        self.conv_block = UNetConvBlock(conv_num, in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out


class ScratchDetectionUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        antialiasing=True,
    ):
        super().__init__()
        self.padding = padding
        self.depth = depth - 1
        prev_channels = in_channels

        self.first = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(in_channels, 2 ** wf, kernel_size=7), nn.LeakyReLU(0.2, True)
        )
        prev_channels = 2 ** wf

        self.down_path = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(depth):
            if antialiasing and depth > 0:
                self.down_sample.append(
                    nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(prev_channels, prev_channels, kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm2d(prev_channels),
                        nn.LeakyReLU(0.2, True),
                        Downsample(channels=prev_channels, stride=2),
                    )
                )
            else:
                self.down_sample.append(
                    nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(prev_channels, prev_channels, kernel_size=4, stride=2, padding=0),
                        nn.BatchNorm2d(prev_channels),
                        nn.LeakyReLU(0.2, True),
                    )
                )
            self.down_path.append(UNetConvBlock(conv_num, prev_channels, 2 ** (wf + i + 1), padding, batch_norm))
            prev_channels = 2 ** (wf + i + 1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(UNetUpBlock(conv_num, prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        if with_tanh:
            self.last = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, out_channels, kernel_size=3), nn.Tanh()
            )
        else:
            self.last = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, out_channels, kernel_size=3))

    def forward(self, x):
        x = self.first(x)
        blocks = []
        for i, down_block in enumerate(self.down_path):
            blocks.append(x)
            x = self.down_sample[i](x)
            x = down_block(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "spectral":
        from torch.nn.utils import spectral_norm
        norm_layer = functools.partial(spectral_norm)
    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")
    return norm_layer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout, dilation)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, dilation):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(dilation)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(dilation)]
        elif padding_type == "zero":
            pass
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(dilation)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(dilation)]
        elif padding_type == "zero":
            pass
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class GlobalGenerator_DCDCv2(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        k_size=3,
        n_downsampling=8,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
        mc=1024,
        start_r=3,
        spatio_size=32,
    ):
        super(GlobalGenerator_DCDCv2, self).__init__()
        activation = nn.ReLU(True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, min(ngf, mc), kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]
        for i in range(start_r):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, mc),
                    min(ngf * mult * 2, mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, mc)),
                activation,
            ]
        for i in range(start_r, n_downsampling - 1):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, mc),
                    min(ngf * mult * 2, mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, mc)),
                activation,
            ]
            model += [ResnetBlock(min(ngf * mult * 2, mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            model += [ResnetBlock(min(ngf * mult * 2, mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        mult = 2 ** (n_downsampling - 1)

        if spatio_size == 32:
            model += [
                nn.Conv2d(
                    min(ngf * mult, mc),
                    min(ngf * mult * 2, mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, mc)),
                activation,
            ]
        if spatio_size == 64:
            model += [ResnetBlock(min(ngf * mult * 2, mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        model += [ResnetBlock(min(ngf * mult * 2, mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.encoder = nn.Sequential(*model)

        model = []
        o_pad = 0 if k_size == 4 else 1
        mult = 2 ** n_downsampling
        model += [
            ResnetBlock(min(ngf * mult, mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        ]

        for i in range(1, n_downsampling - start_r):
            mult = 2 ** (n_downsampling - i)
            model += [
                ResnetBlock(min(ngf * mult, mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer),
                ResnetBlock(min(ngf * mult, mc), padding_type=padding_type, activation=activation, norm_layer=norm_layer),
            ]
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, mc),
                    min(int(ngf * mult / 2), mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), mc)),
                activation,
            ]
        for i in range(n_downsampling - start_r, n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, mc),
                    min(int(ngf * mult / 2), mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), mc)),
                activation,
            ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(min(ngf, mc), output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.decoder = nn.Sequential(*model)

    def forward(self, input, flow="enc_dec"):
        if flow == "enc":
            return self.encoder(input)
        elif flow == "dec":
            return self.decoder(input)
        elif flow == "enc_dec":
            x = self.encoder(input)
            x = self.decoder(x)
            return x


class Mapping_Model(nn.Module):
    def __init__(self, nc, mc=64, n_blocks=3, norm="instance", padding_type="reflect", opt=None):
        super(Mapping_Model, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []
        tmp_nc = 64
        n_up = 4

        for i in range(n_up):
            ic = min(tmp_nc * (2 ** i), mc)
            oc = min(tmp_nc * (2 ** (i + 1)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        
        for i in range(n_blocks):
            model += [
                ResnetBlock(mc, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
            ]

        for i in range(n_up - 1):
            ic = min(64 * (2 ** (4 - i)), mc)
            oc = min(64 * (2 ** (3 - i)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2d(tmp_nc * 2, tmp_nc, 3, 1, 1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Mapping_Model_with_mask(nn.Module):
    def __init__(self, nc, mc=64, n_blocks=3, norm="instance", padding_type="reflect", opt=None):
        super(Mapping_Model_with_mask, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []

        tmp_nc = 64
        n_up = 4

        for i in range(n_up):
            ic = min(tmp_nc * (2 ** i), mc)
            oc = min(tmp_nc * (2 ** (i + 1)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]

        self.before_NL = nn.Sequential(*model)

        if hasattr(opt, 'NL_res') and opt.NL_res:
            self.NL = NonLocalBlock2D_with_mask_Res(
                mc, mc,
                opt.NL_fusion_method if hasattr(opt, 'NL_fusion_method') else 'add',
                opt.correlation_renormalize if hasattr(opt, 'correlation_renormalize') else True,
                opt.softmax_temperature if hasattr(opt, 'softmax_temperature') else 1.0,
                opt.use_self if hasattr(opt, 'use_self') else False,
                opt.cosin_similarity if hasattr(opt, 'cosin_similarity') else False,
            )

        model = []
        for i in range(n_blocks):
            model += [
                ResnetBlock(mc, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
            ]

        for i in range(n_up - 1):
            ic = min(64 * (2 ** (4 - i)), mc)
            oc = min(64 * (2 ** (3 - i)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2d(tmp_nc * 2, tmp_nc, 3, 1, 1)]
        
        self.after_NL = nn.Sequential(*model)
        
    def forward(self, input, mask):
        x1 = self.before_NL(input)
        if hasattr(self, 'NL'):
            x2 = self.NL(x1, mask)
        else:
            x2 = x1
        x3 = self.after_NL(x2)
        return x3


class NonLocalBlock2D_with_mask_Res(nn.Module):
    def __init__(self, in_channels, inter_channels, fusion_method='add', correlation_renormalize=True,
                 temperature=1.0, use_self=False, cosin_similarity=False):
        super(NonLocalBlock2D_with_mask_Res, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.fusion_method = fusion_method
        self.correlation_renormalize = correlation_renormalize
        self.temperature = temperature
        self.use_self = use_self
        self.cosin_similarity = cosin_similarity

        self.g = nn.Conv2d(in_channels, inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, inter_channels, 1)
        self.W = nn.Conv2d(inter_channels, in_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        if self.cosin_similarity:
            theta_x = nn.functional.normalize(theta_x, dim=2)
            phi_x = nn.functional.normalize(phi_x, dim=2)
        
        f = torch.matmul(theta_x, phi_x)
        
        if self.correlation_renormalize:
            f_div_C = f / self.temperature
        else:
            f_div_C = f
        
        f_div_C = nn.functional.softmax(f_div_C, dim=-1)
        
        if mask is not None and mask.numel() > 0:
            mask_flat = mask.view(batch_size, 1, -1)
            f_div_C = f_div_C * mask_flat
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        
        return z


class Pix2PixHDModel_Mapping(nn.Module):
    def __init__(self, opt):
        super(Pix2PixHDModel_Mapping, self).__init__()
        self.opt = opt
        
        ngf = getattr(opt, 'ngf', 64)
        n_downsample_global = getattr(opt, 'n_downsample_global', 3)
        mc = getattr(opt, 'mc', 64)
        k_size = getattr(opt, 'k_size', 4)
        mapping_n_block = getattr(opt, 'mapping_n_block', 6)
        map_mc = getattr(opt, 'map_mc', 512)
        start_r = getattr(opt, 'start_r', 1)
        spatio_size = getattr(opt, 'spatio_size', 32)
        
        norm = getattr(opt, 'norm', 'instance')
        norm_layer = get_norm_layer(norm_type=norm)
        
        self.netG_A = GlobalGenerator_DCDCv2(
            3, 3, ngf, k_size, n_downsample_global, norm_layer, mc=mc, start_r=start_r, spatio_size=spatio_size
        )
        self.netG_B = GlobalGenerator_DCDCv2(
            3, 3, ngf, k_size, n_downsample_global, norm_layer, mc=mc, start_r=start_r, spatio_size=spatio_size
        )
        
        if getattr(opt, 'NL_use_mask', False):
            self.mapping_net = Mapping_Model_with_mask(
                min(ngf * 2 ** n_downsample_global, mc),
                map_mc,
                n_blocks=mapping_n_block,
                opt=opt,
            )
        else:
            self.mapping_net = Mapping_Model(
                min(ngf * 2 ** n_downsample_global, mc),
                map_mc,
                n_blocks=mapping_n_block,
            )
        
        self.mapping_net.apply(weights_init)

    def load_pretrained(self, checkpoint_dir):
        load_pretrainA = getattr(self.opt, 'load_pretrainA', None)
        load_pretrainB = getattr(self.opt, 'load_pretrainB', None)
        load_pretrain = getattr(self.opt, 'load_pretrain', '')
        
        # Fix paths - checkpoint_dir is ../Global/checkpoints, models are in restoration/
        vae_a_path = os.path.join(checkpoint_dir, "restoration", "VAE_A_quality", "latest_net_G.pth")
        vae_b_path_quality = os.path.join(checkpoint_dir, "restoration", "VAE_B_quality", "latest_net_G.pth")
        vae_b_path_scratch = os.path.join(checkpoint_dir, "restoration", "VAE_B_scratch", "latest_net_G.pth")
        vae_b_path_hr = os.path.join(checkpoint_dir, "restoration", "VAE_B_Patch_Attention", "latest_net_G.pth")
        
        mapping_quality_path = os.path.join(checkpoint_dir, "restoration", "mapping_quality", "latest_net_mapping_net.pth")
        mapping_scratch_path = os.path.join(checkpoint_dir, "restoration", "mapping_scratch", "latest_net_mapping_net.pth")
        mapping_hr_path = os.path.join(checkpoint_dir, "restoration", "mapping_Patch_Attention", "latest_net_mapping_net.pth")
        
        if load_pretrainA and os.path.exists(os.path.join(load_pretrainA, "latest_net_G.pth")):
            self.netG_A.load_state_dict(torch.load(os.path.join(load_pretrainA, "latest_net_G.pth"), map_location='cpu'), strict=False)
            print(f"Loaded VAE_A from {load_pretrainA}")
        elif os.path.exists(vae_a_path):
            self.netG_A.load_state_dict(torch.load(vae_a_path, map_location='cpu'), strict=False)
            print(f"Loaded VAE_A from {vae_a_path}")
        
        if load_pretrainB and os.path.exists(os.path.join(load_pretrainB, "latest_net_G.pth")):
            self.netG_B.load_state_dict(torch.load(os.path.join(load_pretrainB, "latest_net_G.pth"), map_location='cpu'), strict=False)
            print(f"Loaded VAE_B from {load_pretrainB}")
        elif os.path.exists(vae_b_path_hr):
            self.netG_B.load_state_dict(torch.load(vae_b_path_hr, map_location='cpu'), strict=False)
            print(f"Loaded VAE_B from {vae_b_path_hr}")
        elif os.path.exists(vae_b_path_scratch):
            self.netG_B.load_state_dict(torch.load(vae_b_path_scratch, map_location='cpu'), strict=False)
            print(f"Loaded VAE_B from {vae_b_path_scratch}")
        elif os.path.exists(vae_b_path_quality):
            self.netG_B.load_state_dict(torch.load(vae_b_path_quality, map_location='cpu'), strict=False)
            print(f"Loaded VAE_B from {vae_b_path_quality}")
        
        if getattr(self.opt, 'NL_use_mask', False):
            if getattr(self.opt, 'mapping_exp', 0) == 1 and os.path.exists(mapping_hr_path):
                self.mapping_net.load_state_dict(torch.load(mapping_hr_path, map_location='cpu'), strict=False)
                print(f"Loaded mapping from {mapping_hr_path}")
            elif os.path.exists(mapping_scratch_path):
                self.mapping_net.load_state_dict(torch.load(mapping_scratch_path, map_location='cpu'), strict=False)
                print(f"Loaded mapping from {mapping_scratch_path}")
        else:
            if os.path.exists(mapping_quality_path):
                self.mapping_net.load_state_dict(torch.load(mapping_quality_path, map_location='cpu'), strict=False)
                print(f"Loaded mapping from {mapping_quality_path}")
        
        self.netG_A.eval()
        self.netG_B.eval()
        if hasattr(self, 'mapping_net'):
            self.mapping_net.eval()

    def to_device(self, device):
        self.netG_A = self.netG_A.to(device)
        self.netG_B = self.netG_B.to(device)
        self.mapping_net = self.mapping_net.to(device)

    def inference(self, label, inst):
        with torch.no_grad():
            label_feat = self.netG_A.forward(label, flow="enc")
            
            if getattr(self.opt, 'NL_use_mask', False):
                # Resize mask to feature resolution
                if inst is not None and inst.numel() > 0:
                    inst_resized = F.interpolate(
                        inst,
                        size=label_feat.shape[2:],
                        mode="nearest"
                    )
                else:
                    inst_resized = inst

                label_feat_map = self.mapping_net(label_feat.detach(), inst_resized)
            else:
                label_feat_map = self.mapping_net(label_feat.detach())
            
            fake_image = self.netG_B.forward(label_feat_map, flow="dec")
        return fake_image


def data_transforms(img, size=256):
    img = img.convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img)
    return img


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Resize(256, interpolation=Image.Resampling.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):
    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255
    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")
    return hole_img


class Args:
    pass


def parameter_set(opt):
    opt.serial_batches = True
    opt.no_flip = True
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    
    if not hasattr(opt, 'checkpoints_dir') or not opt.checkpoints_dir:
        opt.checkpoints_dir = "../Global/checkpoints/"

    if getattr(opt, 'Quality_restore', False):
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "restoration/VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "restoration/VAE_B_quality")
    
    if getattr(opt, 'Scratch_and_Quality_restore', False):
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "restoration/VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "restoration/VAE_B_scratch")
        
        if getattr(opt, 'HR', False):
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"
            opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "restoration/VAE_B_Patch_Attention")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Quality Restoration for Old Photos")
    
    parser.add_argument("--test_input", type=str, required=True, help="Input image directory")
    parser.add_argument("--outputs_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--test_mask", type=str, default="", help="Mask directory for scratched images")
    parser.add_argument("--checkpoints_dir", type=str, default="../Global/checkpoints", help="Checkpoint directory")
    parser.add_argument("--gpu_ids", type=str, default="-1", help="GPU IDs (e.g., 0,1,2 or -1 for CPU/MPS)")
    
    parser.add_argument("--test_mode", type=str, default="Crop", choices=["Scale", "Full", "Crop"], 
                       help="Image processing mode")
    parser.add_argument("--Quality_restore", action="store_true", help="For RGB images without scratches")
    parser.add_argument("--Scratch_and_Quality_restore", action="store_true", help="For scratched images")
    parser.add_argument("--HR", action='store_true', help='Large input size with scratches')
    
    parser.add_argument("--name", type=str, default="", help="Experiment name")
    parser.add_argument("--ngf", type=int, default=64, help="# of gen filters in first conv layer")
    parser.add_argument("--n_downsample_global", type=int, default=3, help="Number of downsampling layers")
    parser.add_argument("--mc", type=int, default=64, help="Max channels")
    parser.add_argument("--k_size", type=int, default=4, help="Kernel size")
    parser.add_argument("--start_r", type=int, default=1, help="Start layer for resblock")
    parser.add_argument("--mapping_n_block", type=int, default=6, help="Number of resblocks in mapping")
    parser.add_argument("--map_mc", type=int, default=512, help="Max channels in mapping")
    parser.add_argument("--norm", type=str, default="instance", help="Normalization type")
    parser.add_argument("--spatio_size", type=int, default=32, help="Spatial size")
    
    parser.add_argument("--mask_dilation", type=int, default=0, help="Mask dilation")
    parser.add_argument("--batchSize", type=int, default=1, help="Batch size")
    parser.add_argument("--load_size", type=int, default=1024, help="Load size")
    parser.add_argument("--fineSize", type=int, default=512, help="Fine size")
    parser.add_argument("--label_nc", type=int, default=35, help="Number of input label channels")
    parser.add_argument("--input_nc", type=int, default=3, help="Number of input image channels")
    parser.add_argument("--output_nc", type=int, default=3, help="Number of output image channels")
    
    parser.add_argument("--no_instance", action="store_true", help="Do not add instance map as input")
    parser.add_argument("--enc_injection", action="store_true", help="Encoder injection")
    parser.add_argument("--dec_injection", action="store_true", help="Decoder injection")
    parser.add_argument("--enc_mask_injection", action="store_true", help="Encoder mask injection")
    parser.add_argument("--dec_mask_injection", action="store_true", help="Decoder mask injection")
    
    parser.add_argument("--which_epoch", type=str, default="latest", help="Which epoch to load")
    parser.add_argument("--phase", type=str, default="test", help="Train, val, test, etc")
    parser.add_argument("--num_test", type=int, default=float("inf"), help="Number of test images")
    
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, mps, cpu")
    
    return parser.parse_args()


def detect_scratches(input_dir, output_dir, device, checkpoint_dir, input_size="full_size"):
    """Detect scratches in images and generate masks automatically."""
    print("\n=== Running Scratch Detection ===")
    
    detection_checkpoint = os.path.join(checkpoint_dir, "detection", "FT_Epoch_latest.pt")
    if not os.path.exists(detection_checkpoint):
        print(f"Warning: Detection model not found at {detection_checkpoint}")
        print("Please download it from the original repository.")
        return None, None
    
    print("Loading scratch detection model...")
    detection_model = ScratchDetectionUNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        antialiasing=True,
    )
    
    checkpoint = torch.load(detection_checkpoint, map_location='cpu')
    if "model_state_dict" in checkpoint:
        detection_model.load_state_dict(checkpoint["model_state_dict"])
    elif "model_state" in checkpoint:
        detection_model.load_state_dict(checkpoint["model_state"])
    else:
        detection_model.load_state_dict(checkpoint)
    
    detection_model = detection_model.to(device)
    detection_model.eval()
    print("Detection model loaded!")
    
    mask_dir = os.path.join(output_dir, "masks")
    mask_output_dir = os.path.join(mask_dir, "mask")
    input_output_dir = os.path.join(mask_dir, "input")
    os.makedirs(mask_output_dir, exist_ok=True)
    os.makedirs(input_output_dir, exist_ok=True)
    
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    print(f"Detecting scratches in {len(input_files)} images...")
    
    for i, image_name in enumerate(input_files):
        image_path = os.path.join(input_dir, image_name)
        if not os.path.isfile(image_path):
            continue
        
        print(f"  [{i+1}/{len(input_files)}] Detecting scratches in {image_name}")
        
        scratch_image = Image.open(image_path).convert("RGB")
        w, h = scratch_image.size
        
        if input_size == "full_size":
            ow, oh = scratch_image.size
            h_out = int(round(oh / 16) * 16)
            w_out = int(round(ow / 16) * 16)
            if (h_out == oh) and (w_out == ow):
                transformed = scratch_image
            else:
                transformed = scratch_image.resize((w_out, h_out), Image.BICUBIC)
        else:
            ow, oh = scratch_image.size
            if ow < oh:
                ow = 256
                oh = int(oh / w * 256)
            else:
                oh = 256
                ow = int(ow / h * 256)
            h_out = int(round(oh / 16) * 16)
            w_out = int(round(ow / 16) * 16)
            transformed = scratch_image.resize((w_out, h_out), Image.BICUBIC)
        
        gray = transformed.convert("L")
        gray_tensor = transforms.ToTensor()(gray)
        gray_tensor = transforms.Normalize([0.5], [0.5])(gray_tensor)
        gray_tensor = gray_tensor.unsqueeze(0).to(device)
        
        _, _, ow, oh = gray_tensor.shape
        if ow < oh:
            ow_scale = 256
            oh_scale = int(oh / ow * 256)
        else:
            oh_scale = 256
            ow_scale = int(ow / oh * 256)
        oh_scale = int(round(oh_scale / 16) * 16)
        ow_scale = int(round(ow_scale / 16) * 16)
        
        gray_scaled = F.interpolate(gray_tensor, [oh_scale, ow_scale], mode="bilinear")
        
        with torch.no_grad():
            P = torch.sigmoid(detection_model(gray_scaled))
        
        P = F.interpolate(P, [h_out, w_out], mode="nearest")
        
        mask_name = os.path.splitext(image_name)[0] + ".png"
        vutils.save_image(
            (P >= 0.4).float(),
            os.path.join(mask_output_dir, mask_name),
            nrow=1, padding=0, normalize=True
        )
        
        transformed.save(os.path.join(input_output_dir, mask_name))
    
    print("=== Scratch Detection Complete ===\n")
    
    return input_output_dir, mask_output_dir


def main():
    args = parse_args()
    
    if args.device == "auto":
        device = get_device()
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = get_device()
    
    print(f"Using device: {device}")

    FIXED_SIZE = 256   # Microsoft default (safe choice)
    
    args.isTrain = False
    
    if not args.Quality_restore and not args.Scratch_and_Quality_restore:
        print("Warning: Neither --Quality_restore nor --Scratch_and_Quality_restore specified.")
        print("Defaulting to --Quality_restore mode.")
        args.Quality_restore = True
    
    parameter_set(args)
    
    if not os.path.exists(args.checkpoints_dir):
        print(f"Error: Checkpoint directory not found: {args.checkpoints_dir}")
        print("Please download the pretrained models first:")
        print("  bash download_models.sh")
        return
    
    os.makedirs(args.outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(args.outputs_dir, "input_image"), exist_ok=True)
    os.makedirs(os.path.join(args.outputs_dir, "restored_image"), exist_ok=True)
    os.makedirs(os.path.join(args.outputs_dir, "origin"), exist_ok=True)
    
    if args.Scratch_and_Quality_restore and not args.test_mask:
        print("\n=== No mask directory provided. Running automatic scratch detection ===")
        input_dir, mask_dir = detect_scratches(
            args.test_input, 
            args.outputs_dir, 
            device, 
            args.checkpoints_dir,
            input_size="full_size"
        )
        if input_dir and mask_dir:
            args.test_mask = mask_dir
            args.test_input_for_restore = input_dir
        else:
            print("Error: Scratch detection failed. Please provide mask directory manually.")
            return
    else:
        args.test_input_for_restore = args.test_input
    
    print("Initializing model...")
    model = Pix2PixHDModel_Mapping(args)
    model.load_pretrained(args.checkpoints_dir)
    model.to_device(device)
    print("Model loaded successfully!")
    
    input_files = [f for f in os.listdir(args.test_input_for_restore) if os.path.isfile(os.path.join(args.test_input_for_restore, f))]
    input_files.sort()
    
    mask_files = []
    if args.test_mask:
        mask_files = [f for f in os.listdir(args.test_mask) if os.path.isfile(os.path.join(args.test_mask, f))]
        mask_files.sort()
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.ToTensor()
    
    print(f"Processing {len(input_files)} image(s)...")
    
    for i, input_name in enumerate(input_files):
        input_file = os.path.join(args.test_input_for_restore, input_name)
        if not os.path.isfile(input_file):
            print(f"Skipping non-file {input_name}")
            continue
        
        print(f"Processing [{i+1}/{len(input_files)}]: {input_name}")
        
        try:
            input_img = Image.open(input_file).convert("RGB")
            origin = input_img.copy()
            
            if getattr(args, 'NL_use_mask', False):
                if i >= len(mask_files):
                    print(f"  Warning: No mask found for {input_name}, skipping")
                    continue
                
                mask_name = mask_files[i]
                mask = Image.open(os.path.join(args.test_mask, mask_name)).convert("RGB")
                
                if args.mask_dilation != 0:
                    kernel = np.ones((3, 3), np.uint8)
                    mask_np = np.array(mask)
                    mask_np = cv2.dilate(mask_np, kernel, iterations=args.mask_dilation)
                    mask = Image.fromarray(mask_np.astype('uint8'))
                
                input_img = irregular_hole_synthesize(input_img, mask)
                
                # Ensure input_img is also aligned to 4 pixels
                input_img = data_transforms(input_img, scale=False)
                
                mask = mask_transform(mask)
                mask = mask[:1, :, :]
                mask = mask.unsqueeze(0).to(device)
                
                input_tensor = img_transform(input_img)
                input_tensor = input_tensor.unsqueeze(0).to(device)
            else:
                if args.test_mode == "Scale":
                    input_img = data_transforms(input_img, scale=True)
                elif args.test_mode == "Full":
                    input_img = data_transforms(input_img, scale=False)
                elif args.test_mode == "Crop":
                    input_img = data_transforms_rgb_old(input_img)
                
                input_tensor = img_transform(input_img)
                input_tensor = input_tensor.unsqueeze(0).to(device)
                mask = torch.zeros_like(input_tensor)
            
            generated = model.inference(input_tensor, mask)
            
            if input_name.endswith(".jpg"):
                input_name = input_name[:-4] + ".png"
            
            vutils.save_image(
                (input_tensor + 1.0) / 2.0,
                os.path.join(args.outputs_dir, "input_image", input_name),
                nrow=1, padding=0, normalize=True
            )
            vutils.save_image(
                (generated.cpu() + 1.0) / 2.0,
                os.path.join(args.outputs_dir, "restored_image", input_name),
                nrow=1, padding=0, normalize=True
            )
            origin.save(os.path.join(args.outputs_dir, "origin", input_name))
            
            print(f"  Saved: {input_name}")
            
        except Exception as e:
            print(f"  Error processing {input_name}: {e}")
            continue
    
    print("\nDone! All images processed.")
    print(f"Results saved to: {args.outputs_dir}")


if __name__ == "__main__":
    main()
