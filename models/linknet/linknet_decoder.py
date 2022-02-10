import torch
import torch.nn as nn
from ..base import modules
from ..base.modules import Activation

class TransposeX2(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))

        super().__init__(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential(
            modules.Conv2dReLU(in_channels, in_channels // 4, kernel_size=1, use_batchnorm=use_batchnorm),
            TransposeX2(in_channels // 4, in_channels // 4, use_batchnorm=use_batchnorm),
            modules.Conv2dReLU(in_channels // 4, out_channels, kernel_size=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x, skip=None):
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x

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


class LinknetDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            prefinal_channels=32,
            n_blocks=5,
            use_batchnorm=True,
            seg_in_channels=None,
            seg_out_channels=None,
            seg_activation=None,
            seg_kernel_size=None,
            seg_att_depth=None
    ):
        super().__init__()

        encoder_channels = encoder_channels[1:]  # remove first skip
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        channels = list(encoder_channels) + [prefinal_channels]

        self.blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i + 1], use_batchnorm=use_batchnorm)
            for i in range(n_blocks)
        ])

        self.seg_in_channels = seg_in_channels
        self.seg_out_channels = seg_out_channels
        self.seg_activation = seg_activation
        self.seg_kernel_size = seg_kernel_size
        self.seg_att_depth = seg_att_depth

        self.segmentation = SegmentationHead(in_channels=self.seg_in_channels, out_channels=self.seg_out_channels,
                                             kernel_size=self.seg_kernel_size, activation=self.seg_activation,
                                             att_depth=self.seg_att_depth)


    def forward(self, features, attentions):
            features = features[1:]  # remove first skip
            features = features[::-1]  # reverse channels to start from head of encoder

            x = features[0]
            skips = features[1:]

            for i, decoder_block in enumerate(self.blocks):
                skip = skips[i] if i < len(skips) else None
                x = decoder_block(x, skip)

            x, attentions = self.segmentation(x, attentions)

            return x, attentions

