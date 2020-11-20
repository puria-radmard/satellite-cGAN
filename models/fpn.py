from imports import *
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

# https://github.com/qubvel/segmentation_models.pytorch/

class SegmentationHead(nn.Sequential):

    def __init__(self, n_channels, n_classes, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(n_channels, n_classes, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


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
    def __init__(self, n_channels, n_classes, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(n_channels, n_classes, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(n_classes, n_classes, upsample=True))

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


class Conv3x3GNReLU(nn.Module):
    def __init__(self, n_channels, n_classes, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                n_channels, n_classes, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, n_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
    ):
        super().__init__()

        self.n_classes = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
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

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x

class ResNetEncoder(ResNet):
    def __init__(self, encoder_name, n_channels, depth, **kwargs):

        params = resnet_encoders[encoder_name]["params"]
        self._n_classes = params["n_classes"]
        self._depth = depth

        super().__init__(block = params["block"], layers = params["layers"])

        del self.fc
        del self.avgpool

        if n_channels != 3:

            self.n_channels = n_channels
            if self._n_classes[0] == 3:
                self._n_classes = tuple([n_channels] + list(self._n_classes)[1:])

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
    def __init__(self, n_channels, n_classes, kernel_size=3, upsampling=1):

        super(FPNOutBlock, self).__init__()

        conv2d = nn.Conv2d(
            n_channels, n_classes, kernel_size=kernel_size, padding=kernel_size // 2
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
            encoder_channels=self.encoder._n_classes,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=dropout,
            merge_policy=decoder_merge_policy,
        )

        self.outblock = SegmentationHead(
            n_channels=self.decoder.n_classes,
            n_classes=n_classes,
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
            "n_classes": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x4d": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "encoder": ResNetEncoder,
        "params": {
            "n_classes": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}
