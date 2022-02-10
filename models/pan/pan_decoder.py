import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.modules import Activation

class ConvBnRelu(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            add_relu: bool = True,
            interpolate: bool = False
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups
        )
        self.add_relu = add_relu
        self.interpolate = interpolate
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPABlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            upscale_mode='bilinear'
    ):
        super(FPABlock, self).__init__()

        self.upscale_mode = upscale_mode
        if self.upscale_mode == 'bilinear':
            self.align_corners = True
        else:
            self.align_corners = False

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )

        # midddle branch
        self.mid = nn.Sequential(
            ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=in_channels, out_channels=1, kernel_size=7, stride=1, padding=3)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        upscale_parameters = dict(
            mode=self.upscale_mode,
            align_corners=self.align_corners
        )
        b1 = F.interpolate(b1, size=(h, w), **upscale_parameters)

        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), **upscale_parameters)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), **upscale_parameters)

        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), **upscale_parameters)

        x = torch.mul(x, mid)
        x = x + b1
        return x


class GAUBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            upscale_mode: str = 'bilinear'
    ):
        super(GAUBlock, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = True if upscale_mode == 'bilinear' else None

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=1, add_relu=False),
            nn.Sigmoid()
        )
        self.conv2 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x, y):
        """
        Args:
            x: low level feature
            y: high level feature
        """
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(
            y, size=(h, w), mode=self.upscale_mode, align_corners=self.align_corners
        )
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)
        return y_up + z


class PANDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            upscale_mode: str = 'bilinear',
            seg_in_channels=None,
            seg_out_channels=None,
            seg_activation=None,
            seg_kernel_size=None,
            seg_upsampling=3,
            seg_att_depth=None
    ):
        super().__init__()

        self.fpa = FPABlock(in_channels=encoder_channels[-1], out_channels=decoder_channels)
        self.gau3 = GAUBlock(in_channels=encoder_channels[-2], out_channels=decoder_channels, upscale_mode=upscale_mode)
        self.gau2 = GAUBlock(in_channels=encoder_channels[-3], out_channels=decoder_channels, upscale_mode=upscale_mode)
        self.gau1 = GAUBlock(in_channels=encoder_channels[-4], out_channels=decoder_channels, upscale_mode=upscale_mode)

        self.seg_in_channels = seg_in_channels
        self.seg_out_channels = seg_out_channels
        self.seg_activation = seg_activation
        self.seg_kernel_size = seg_kernel_size
        self.seg_upsampling = seg_upsampling
        self.seg_att_depth = seg_att_depth

        self.segmentation = SegmentationHead(in_channels=self.seg_in_channels, out_channels=self.seg_out_channels, kernel_size=self.seg_kernel_size, activation=self.seg_activation,upsampling=self.seg_upsampling, att_depth=self.seg_att_depth)

    def forward(self, features, attentions):
        bottleneck = features[-1]
        x5 = self.fpa(bottleneck)         # 1/32
        x4 = self.gau3(features[-2], x5)  # 1/16
        x3 = self.gau2(features[-3], x4)  # 1/8
        x2 = self.gau1(features[-4], x3)  # 1/4

        x, attentions=self.segmentation(x2,attentions)

        return x, attentions


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, patch_size=10, activation=None, upsampling=1, att_depth=3):
        super().__init__()
        self.patch_size=patch_size

        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        self.out_channels=out_channels

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        self.activation = Activation(activation)
        self.att_depth = att_depth

    def forward(self, x, attentions):

        conv_feamap_size = nn.Conv2d(self.out_channels,self.out_channels, kernel_size=(2**self.att_depth, 2**self.att_depth),stride=(2**self.att_depth, 2**self.att_depth),groups=self.out_channels,bias=False)
        conv_feamap_size.weight=nn.Parameter(torch.ones((self.out_channels, 1, 2**self.att_depth, 2**self.att_depth)))
        conv_feamap_size.to(x.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        x = self.conv_x(x)
        x = self.upsampling(x)
        fold_layer = torch.nn.Fold(output_size=(x.size()[-2], x.size()[-1]), kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        correction=[]

        x_argmax=torch.argmax(x, dim=1)

        pr_temp = torch.zeros(x.size()).to(x.device)
        src = torch.ones(x.size()).to(x.device)
        x_softmax = pr_temp.scatter(dim=1, index=x_argmax.unsqueeze(1), src=src)

        argx_feamap = conv_feamap_size(x_softmax) / (2 ** self.att_depth * 2 ** self.att_depth)


        for i in range(x.size()[1]):
            non_zeros = torch.unsqueeze(torch.count_nonzero(attentions[:, i:i + 1, :, :], dim=-1) + 0.00001,dim=-1)

            att = torch.matmul(attentions[:,i:i + 1, :, :]/non_zeros, torch.unsqueeze(self.unfold(argx_feamap[:, i:i + 1, :, :]), dim=1).transpose(-1, -2))

            att=torch.squeeze(att, dim=1)

            att = fold_layer(att.transpose(-1, -2))

            correction.append(att)

        correction=torch.cat(correction, dim=1)

        x = correction * x + x

        x = self.activation(x)

        return x, attentions