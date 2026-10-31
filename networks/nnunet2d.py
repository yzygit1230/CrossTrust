import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization_layer(channels, mode):
    if mode == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if mode == "batch":
        return nn.BatchNorm2d(channels)
    if mode == "group":
        groups = min(16, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if mode == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {mode}")


class PlainConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm="instance", dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            normalization_layer(out_channels, norm),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend(
            [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                normalization_layer(out_channels, norm),
                nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm="instance", dropout=0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalization_layer(out_channels, norm),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            PlainConvBlock(out_channels, out_channels, norm=norm, dropout=dropout),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, norm="instance"):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.conv = PlainConvBlock(
            out_channels + skip_channels,
            out_channels,
            norm=norm,
            dropout=0.0,
        )

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class NNUNet2D(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        base_channels=32,
        max_channels=320,
        depth=5,
        norm="instance",
        dropout=0.0,
    ):
        super().__init__()
        if depth < 3:
            raise ValueError("NNUNet2D depth must be at least 3.")
        channels = [
            min(base_channels * (2**level), max_channels)
            for level in range(depth)
        ]
        self.stem = PlainConvBlock(
            n_channels,
            channels[0],
            norm=norm,
            dropout=dropout,
        )
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    channels[level - 1],
                    channels[level],
                    norm=norm,
                    dropout=dropout if level == depth - 1 else 0.0,
                )
                for level in range(1, depth)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    channels[level],
                    channels[level - 1],
                    channels[level - 1],
                    norm=norm,
                )
                for level in range(depth - 1, 0, -1)
            ]
        )
        self.out = nn.Conv2d(channels[0], n_classes, kernel_size=1)

    def forward(self, x, feature=False):
        skips = [self.stem(x)]
        for block in self.down_blocks:
            skips.append(block(skips[-1]))

        x = skips[-1]
        for block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = block(x, skip)
        logits = self.out(x)
        if feature:
            return logits, x
        return logits
