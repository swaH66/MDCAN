"""MDCAN model definition.

The implementation names follow the manuscript: DCHA, ESA, MSFB, MSCA, and
MDCAN. Parameter-bearing attribute paths are stable so released MDCAN state
dictionaries can be loaded with ``strict=True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ECA(nn.Module):
    """Efficient cross-channel attention used by DCHA."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels)) / gamma) + b))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        weights = self.avg_pool(x).view(batch_size, 1, channels)
        weights = self.conv(weights).view(batch_size, channels, 1, 1)
        return self.sigmoid(weights)


class DCHA(nn.Module):
    """Dual-Context Hybrid Attention (DCHA)."""

    def __init__(self, in_channels: int, reduction: int = 16, use_eca: bool = True) -> None:
        super().__init__()
        self.use_eca = use_eca
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(8, in_channels // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
        )
        if use_eca:
            self.eca = ECA(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_descriptor = self.shared_mlp(self.max_pool(x))
        avg_descriptor = self.shared_mlp(self.avg_pool(x))
        global_channel_attention = self.sigmoid(max_descriptor + avg_descriptor)
        if not self.use_eca:
            return global_channel_attention
        local_channel_attention = self.eca(x)
        return self.sigmoid(global_channel_attention + local_channel_attention)


class LNLA(nn.Module):
    """Lightweight non-local attention used inside ESA."""

    def __init__(self, scale_factor: int = 8) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        reduce_spatial_size = height > 32 or width > 32
        if reduce_spatial_size:
            reduced = F.interpolate(
                x,
                scale_factor=1.0 / self.scale_factor,
                mode="bilinear",
                align_corners=False,
            )
        else:
            reduced = x

        _, _, reduced_height, reduced_width = reduced.shape
        query = reduced.view(batch_size, -1, reduced_height * reduced_width).permute(0, 2, 1)
        key = reduced.view(batch_size, -1, reduced_height * reduced_width)
        affinity = self.softmax(torch.bmm(query, key))
        attention = torch.mean(affinity, dim=1, keepdim=True)
        attention = attention.view(batch_size, 1, reduced_height, reduced_width)

        if reduce_spatial_size:
            attention = F.interpolate(
                attention, size=(height, width), mode="bilinear", align_corners=False
            )
        return attention


class ESA(nn.Module):
    """Efficient Spatial Attention (ESA)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.non_local = LNLA()
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_map = torch.max(x, dim=1, keepdim=True).values
        average_map = torch.mean(x, dim=1, keepdim=True)
        local_spatial_map = self.conv(torch.cat([max_map, average_map], dim=1))
        non_local_map = self.non_local(x)
        return self.sigmoid(local_spatial_map + self.gamma * non_local_map)


class DSConv(nn.Module):
    """Depthwise-separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class DCB(nn.Module):
    """Dilated depthwise-separable convolution branch."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = DSConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class GPP(nn.Module):
    """Global pooling branch used by MSFB."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        feature = self.relu(self.bn(self.conv(self.pool(x))))
        return F.interpolate(feature, size=(height, width), mode="bilinear", align_corners=True)


class MSFB(nn.Module):
    """Multi-Scale Feature Block (MSFB)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        adjusted_channels = (in_channels // 4) * 4
        self.ch_per_group = adjusted_channels // 4
        self.conv1x1 = self._branch(nn.Conv2d(in_channels, self.ch_per_group, 1, bias=False))
        self.conv3x3 = self._branch(
            DSConv(in_channels, self.ch_per_group, kernel_size=3, padding=1, bias=False)
        )
        self.conv5x5 = self._branch(
            DSConv(in_channels, self.ch_per_group, kernel_size=5, padding=2, bias=False)
        )
        self.dilated_conv = nn.Sequential(DCB(in_channels, self.ch_per_group, dilation=2))
        self.ppm = GPP(in_channels, self.ch_per_group)
        self.fusion_conv = nn.Conv2d(
            adjusted_channels + self.ch_per_group, in_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU6(inplace=True)
        self.layer_norm = nn.LayerNorm([in_channels, 7, 7])
        self.dropblock = nn.Dropout2d(0.1)

    @staticmethod
    def _branch(convolution: nn.Module) -> nn.Sequential:
        out_channels = convolution.pointwise.out_channels if isinstance(convolution, DSConv) else convolution.out_channels
        return nn.Sequential(convolution, nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        branches = [
            self.conv1x1(x),
            self.conv3x3(x),
            self.conv5x5(x),
            self.dilated_conv(x),
            self.ppm(x),
        ]
        fused = self._channel_shuffle(torch.cat(branches, dim=1))

        # Kept identical to the implementation used in the reported experiments.
        fused = F.batch_norm(fused, None, None, None, None, True, 0.1, 1e-5)
        output = self.bn(self.fusion_conv(fused))
        if output.shape[-2:] == (7, 7):
            output = self.layer_norm(output)
        output = self.dropblock(self.relu(output))
        return output + identity

    @staticmethod
    def _channel_shuffle(x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        groups = 5
        if channels % groups != 0:
            raise ValueError(f"MSFB channel count ({channels}) must be divisible by {groups}.")
        x = x.view(batch_size, groups, channels // groups, height, width)
        return x.transpose(1, 2).contiguous().view(batch_size, channels, height, width)


class MSCA(nn.Module):
    """Multi-Scale Cascaded Attention (MSCA): DCHA -> ESA -> MSFB."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.dcha = DCHA(in_channels, reduction, use_eca=True)
        self.esa = ESA(kernel_size=7)
        self.msfb = MSFB(in_channels)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # Retained because it is part of the released model state dictionary.
        self.class_weight_attn = nn.Parameter(torch.ones(8))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        channel_refined = x * self.dcha(x)
        spatial_refined = channel_refined * self.esa(channel_refined)
        multi_scale_feature = self.msfb(spatial_refined)
        output = self.bn(self.final_conv(multi_scale_feature))
        return self.relu(output + self.gamma * identity)


class MDCAN(nn.Module):
    """Multi-scale Dual-path Cascaded Attention Network (MDCAN)."""

    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet = models.densenet121(weights=weights)
        self.densenet.classifier = nn.Identity()
        self.features = self.densenet.features
        self.feature_channels = 1024

        # These attribute names form part of the released state-dict interface.
        self.global_path = nn.Sequential(nn.AdaptiveAvgPool2d(7), MSCA(self.feature_channels))
        self.local_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), MSCA(self.feature_channels)
        )
        self.path_fusion = nn.Conv2d(self.feature_channels * 2, self.feature_channels, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._init_added_weights()

    def _init_added_weights(self) -> None:
        # This preserves the initialization behavior of the original code.
        for module in [self.path_fusion, self.classifier]:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone_feature = self.features(x)
        global_feature = self.global_path(backbone_feature)
        local_feature = self.local_path(backbone_feature)
        fused_feature = self.path_fusion(torch.cat([global_feature, local_feature], dim=1))
        pooled_feature = torch.flatten(self.global_pool(fused_feature), 1)
        return self.classifier(pooled_feature)


def create_mdcan(pretrained: bool = True, num_classes: int = 8) -> MDCAN:
    """Create an MDCAN model."""
    return MDCAN(num_classes=num_classes, pretrained=pretrained)

__all__ = ["DCHA", "ESA", "MSFB", "MSCA", "MDCAN", "create_mdcan"]
