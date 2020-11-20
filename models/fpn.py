from imports import *
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

# https://github.com/qubvel/segmentation_models.pytorch/


class ResNetEncoder(ResNet):
    def __init__(self, encoder_name, n_channels, depth, **kwargs):

        params = resnet_encoders[encoder_name]["params"]
        self._out_channels = params["out_channels"]
        self._depth = depth

        super().__init__(block = params["block"], layers = params["layers"])

        del self.fc
        del self.avgpool

        if n_channels != 3:

            self.n_channels = n_channels
            if self._out_channels[0] == 3:
                self._out_channels = tuple([n_channels] + list(self._out_channels)[1:])

            utils.patch_first_conv(model=self, n_channels=n_channels)

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


class FPNOutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):

        super(FPNOutBlock, self).__init__()

        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )

        self.model = nn.Sequential([conv2d, upsampling])

    def forward(self, X):

        return self.model(X)


class FPN(nn.Module):
    def __init__(
        self, dropout, sigmoid_channels, resnet_encoder, n_channels=3, n_classes=1
    ):

        super(FPN, self).__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.sigmoid_channels = sigmoid_channels

        encoder_depth = 5
        decoder_pyramid_channels = 256
        decoder_segmentation_channels = 128
        decoder_merge_policy = "add"  # or "cat"
        upsampling = 4

        if len(sigmoid_channels) != n_classes:
            raise ValueError("len(sigmoid_channels) != n_classes")

        self.encoder = ResNetEncoder(
            encoder_name=resnet_encoder, n_channels=n_channels, depth=encoder_depth
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=dropout,
            merge_policy=decoder_merge_policy,
        )

        self.outblock = SegmentationHead(
            n_channels=self.decoder.out_channels,
            out_channels=n_classes,
            kernel_size=1,
            upsampling=upsampling,
        )

        # removed self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.outblock(decoder_output)

        if len(self.sigmoid_channels) != 1:
            raise NotImplementedError("Haven't figured out mixed result yet")

        if self.sigmoid_channels[0]:
            return nn.Sigmoid()(logits)
        else:
            return logits


resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x4d": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "encoder": ResNetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}
