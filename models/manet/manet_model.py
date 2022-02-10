import torch
from typing import Optional, Union, List
from .manet_decoder import MAnetDecoder
from ..base import get_encoder, ClassificationHead
from ..base import initialization as init

class My_MAnet(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_pab_channels: int = 64,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        att_depth=99
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            att_depth=att_depth
        )

        self.decoder = MAnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            pab_channels=decoder_pab_channels,
            in_channels=decoder_channels[-1],
            out_classes=classes,
            activation=activation,
            kernel_size=3,
            att_depth=att_depth
        )


        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "manet-{}".format(encoder_name)
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
