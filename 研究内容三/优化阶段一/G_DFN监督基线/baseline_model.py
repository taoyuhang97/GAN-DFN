# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SparseInstanceBaselineUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        slots_per_voxel: int,
        base_channels: int = 8,
        dx: float = 12.5,
        dy: float = 12.5,
        dz: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.slots_per_voxel = int(slots_per_voxel)
        self.base_channels = int(base_channels)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        self.enc1 = ConvBlock3D(self.in_channels, c1)
        self.enc2 = DownBlock3D(c1, c2)
        self.enc3 = DownBlock3D(c2, c3)
        self.enc4 = DownBlock3D(c3, c4)

        self.dec3 = UpBlock3D(c4, c3, c3)
        self.dec2 = UpBlock3D(c3, c2, c2)
        self.dec1 = UpBlock3D(c2, c1, c1)

        self.count_head = nn.Conv3d(c1, 1, kernel_size=1)
        self.center_head = nn.Conv3d(c1, self.slots_per_voxel, kernel_size=1)
        self.geom_head = nn.Conv3d(c1, self.slots_per_voxel * 12, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        count_pred = F.softplus(self.count_head(d1))
        center_logits = self.center_head(d1)
        geom_raw = self.geom_head(d1)

        batch_size, _, nx, ny, nz = geom_raw.shape
        geom_raw = geom_raw.view(batch_size, self.slots_per_voxel, 12, nx, ny, nz)

        offset_x = torch.tanh(geom_raw[:, :, 0:1]) * (self.dx / 2.0)
        offset_y = torch.tanh(geom_raw[:, :, 1:2]) * (self.dy / 2.0)
        offset_z = torch.tanh(geom_raw[:, :, 2:3]) * (self.dz / 2.0)

        normals = F.normalize(geom_raw[:, :, 3:6], dim=2, eps=1e-6)
        u_dirs = geom_raw[:, :, 6:9]
        projection = (u_dirs * normals).sum(dim=2, keepdim=True)
        u_dirs = F.normalize(u_dirs - projection * normals, dim=2, eps=1e-6)

        length = F.softplus(geom_raw[:, :, 9:10]) + 1e-6
        height = F.softplus(geom_raw[:, :, 10:11]) + 1e-6
        confidence = torch.sigmoid(geom_raw[:, :, 11:12])

        geom_pred = torch.cat(
            [offset_x, offset_y, offset_z, normals, u_dirs, length, height, confidence],
            dim=2,
        )
        return {
            "count_pred": count_pred,
            "center_logits": center_logits,
            "geom_pred": geom_pred,
        }
