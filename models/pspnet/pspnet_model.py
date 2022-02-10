import torch
from typing import Optional, Union
from .pspnet_decoder import PSPDecoder
from ..base import get_encoder
from ..base import initialization as init

class My_pspnet(torch.nn.Module):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_depth: int = 5,

        psp_out_channels: int = 512,

        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 8,
        att_depth: int = 99,
        replace_stride_with_dilation=[False, True, True]
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            att_depth=att_depth,
            replace_stride_with_dilation=replace_stride_with_dilation
        )

        self.decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            psp_out_channels=psp_out_channels,
            dropout=psp_dropout,
            seg_in_channels=psp_out_channels,
            seg_out_channels=classes,
            seg_activation=activation,
            seg_kernel_size=3,
            seg_upsampling=upsampling,
            seg_att_depth=att_depth
        )

        self.name = "psp-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features, attentions = self.encoder(x)

        output = self.decoder(features,attentions)

        return output

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
        x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
        prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
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

