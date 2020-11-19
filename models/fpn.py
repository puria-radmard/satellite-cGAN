from imports import *

# https://github.com/qubvel/segmentation_models.pytorch/

class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

class FPN(nn.Module):

    def __init__(self, dropout, sigmoid_channels, n_channels=3, n_classes=1):

        super(FPN, self).__init__()

        self.no_skips = no_skips
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.sigmoid_channels = sigmoid_channels

        if len(sigmoid_channels) != n_classes:
            raise ValueError("len(sigmoid_channels) != n_classes")

    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        dropout: spatial dropout rate in range (0, 1).

        upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)
    Returns:
        ``torch.nn.Module``: **FPN**
    """

    def __init__(
        self,
        
        
        dropout: float = 0.2,
        sigmoid_channels
        n_channels: int = 3,
        n_classes: int = 1,
        
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            n_channels=n_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = SegmentationHead(
            n_channels=self.decoder.out_channels,
            out_channels=n_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

        encoder_depth = 5
        decoder_pyramid_channels = 256
        decoder_segmentation_channels = 128
        decoder_merge_policy = "add"     # or "cat"
        upsampling = 4