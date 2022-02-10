from typing import Optional, Union
from .fpn_decoder import FPNDecoder
import torch
from ..base import get_encoder
from ..base import initialization as init
from ..base import SegmentationHead, ClassificationHead


class My_FPN(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        att_depth: int = 99
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            att_depth= att_depth
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
            seg_in_channels=128,
            seg_out_channels=classes,
            seg_activation=activation,
            seg_kernel_size=1,
            seg_upsampling=upsampling,
            seg_att_depth=att_depth

        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "FPN-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        # init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        features, attentions = self.encoder(x)
        output, attentions = self.decoder(features, attentions)

        return output, attentions

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x, attentions = self.forward(x)

        return x

    def present(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x, attentions = self.forward(x)

        return x, attentions
