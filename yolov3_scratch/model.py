import torch
from torch import nn


config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    # first route from the end of previous block
    (512, 3, 2),
    ["B", 8],
    # second route from the end of previous block
    (1024, 3, 2),
    ["B", 4],
    # Until here is YOLO-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S"
]

class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act = True, **kwargs):
        super().__init__()
        # Bias is unnecessary if we are using BatchNorm
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            x = self.conv(x)
            x = self.bn(x)
            return self.leaky(x)
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CnnBlock(channels, channels // 2, kernel_size=1),
                    CnnBlock(channels // 2, channels, kernel_size=3, padding=1)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CnnBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            # 3 anchors for each grid cell, num_classes times score, 5 for the (objectness_score, x, y, w, h)
            CnnBlock(2*in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1)
        )
        self.num_classes = num_classes

    def forward(self, x):
        # X could be example (batch_size, 3, 13, 13) -> (batch_size, 3, 13, 13, num_class*5)
        return self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)


class Yolov3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CnnBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=1 if kernel_size == 3 else 0))
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            elif isinstance(module, str):
                if module == "S": # Scale prediction
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CnnBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module == "U": # Upsample
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers

    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs


if __name__ == '__main__':
    num_classes = 20
    IMAGE_SIZE = 416
    model = Yolov3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    print("")