import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.modules import Activation


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


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

class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
            # 以下是segmentation head的参数
            seg_in_channels=None,
            seg_out_channels=None,
            seg_activation=None,
            seg_kernel_size=None,
            seg_upsampling=None,
            seg_att_depth=None
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [3, 2, 1, 0]
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        self.seg_in_channels = seg_in_channels
        self.seg_out_channels = seg_out_channels
        self.seg_activation = seg_activation
        self.seg_kernel_size = seg_kernel_size
        self.seg_upsampling=seg_upsampling
        self.seg_att_depth = seg_att_depth
        self.segmentation = SegmentationHead(in_channels=self.seg_in_channels, out_channels=self.seg_out_channels,
                                             kernel_size=self.seg_kernel_size, activation=self.seg_activation,upsampling=self.seg_upsampling,
                                             att_depth=self.seg_att_depth)

    def forward(self, features, attentions):
        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        x, attentions = self.segmentation(x, attentions)
        return x, attentions
