import torch
from typing import Optional, Union
from .pan_decoder import PANDecoder
from ..base import get_encoder
from ..base import initialization as init
from ..base import SegmentationHead, ClassificationHead

class My_PAN(torch.nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: Optional[str] = "imagenet",
            encoder_dilation: bool = True,
            decoder_channels: int = 32,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,
            att_depth: int=99

    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
            att_depth=att_depth
        )

        if encoder_dilation:
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )

        self.decoder = PANDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            seg_in_channels=decoder_channels,
            seg_out_channels=classes,
            seg_activation=activation,
            seg_kernel_size=3,
            seg_upsampling=upsampling,
            seg_att_depth=att_depth
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)


    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features, attentions = self.encoder(x)
        output, attentions= self.decoder(features, attentions)

        return output, attentions

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
