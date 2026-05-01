# ------------------------------------------------------------
# 作用：模型训练及物测井区域体素表示生成
# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
3D-GAN 示例：地震块 -> 裂缝 voxel 5 通道
输入: [B, 1, 24, 24, Z]
输出: [B, 5, 24, 24, Z]
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pyvista as pv
import re
import torch.nn.functional as F

# ------------------------------
# 数据集
# ------------------------------
def parse_xy_from_filename(filename):
    match = re.search(r'_X(\d+)_Y(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        raise ValueError(f"文件名中未找到 XY 信息：{filename}")

def corners_to_voxel_centers(corner_data):
    """
    corner_data: np.array, shape = [nx+1, ny+1, nz+1]
    return: np.array, shape = [nx, ny, nz]
    """
    voxel_centers = (
        corner_data[:-1, :-1, :-1] + corner_data[1:, :-1, :-1] +
        corner_data[:-1, 1:, :-1] + corner_data[1:, 1:, :-1] +
        corner_data[:-1, :-1, 1:] + corner_data[1:, :-1, 1:] +
        corner_data[:-1, 1:, 1:] + corner_data[1:, 1:, 1:]
    ) / 8.0
    return voxel_centers


class SeismicVoxelDataset(Dataset):
    def __init__(self, seismic_dir, voxel_in):
        self.voxel_in = voxel_in
        self.seismic_dir = seismic_dir

        # 所有体素文件
        self.voxel_files = sorted([f for f in os.listdir(voxel_in) if f.endswith(".npy")])
        self.seismic_files = []

        for vf in self.voxel_files:
            # 从 voxel 文件名提取 X,Y
            m = re.search(r"X(\d+)_Y(\d+)", vf)
            if m:
                vx, vy = m.groups()
                # 寻找对应的地震文件
                sf_name = f"sample_X{vx}_Y{vy}.npy"
                sf_path = os.path.join(seismic_dir, sf_name)
                if os.path.exists(sf_path):
                    self.seismic_files.append(sf_path)
                else:
                    raise FileNotFoundError(f"未找到对应地震文件: {sf_path}")
            else:
                raise ValueError(f"体素文件名格式错误: {vf}")

        # 完整 voxel 文件路径
        self.voxel_files = [os.path.join(voxel_in, vf) for vf in self.voxel_files]

        assert len(self.seismic_files) == len(self.voxel_files), "匹配后数量不一致"

    def __len__(self):
        return len(self.voxel_files)

    def __getitem__(self, idx):
        # 加载 voxel 中心
        y = np.load(self.voxel_files[idx]).astype(np.float32)    # [24,24,2700,5]

        # 加载对应 seismic
        # 根据 voxel 文件名提取 X/Y
        name = os.path.basename(self.voxel_files[idx])
        parts = name.replace(".npy","").split("_")
        key = f"{parts[2]}_{parts[3]}"  # 'X14_Y21'

        # 找到对应 seismic 文件
        matched_s = [s for s in self.seismic_files if key in os.path.basename(s)]
        assert len(matched_s) == 1
        x_corner = np.load(matched_s[0]).astype(np.float32)  # [25,25,2701,1]

        # 转换为 voxel center
        x_center = corners_to_voxel_centers(x_corner[...,0])[...,None]  # [24,24,2700,1]

        # 转 torch
        x = torch.from_numpy(x_center).permute(3,0,1,2)  # [1,24,24,2700]
        y = torch.from_numpy(y).permute(3,0,1,2)         # [5,24,24,2700]

        return x, y


# ------------------------------
# 3D-GAN Generator
# ------------------------------
def match_target_z(x, target_z=2700):
    z = x.shape[-1]
    if z == target_z:
        return x
    elif z < target_z:
        pad = target_z - z
        return F.pad(x, (0, pad))   # (left=0, right=pad)
    else:
        return x[..., :target_z]    # 裁剪到 2700

class Generator3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=5, base=32):
        super().__init__()
        self.net = nn.Sequential(
            # 编码
            nn.Conv3d(in_ch, base, 4, 2, 1), nn.BatchNorm3d(base), nn.ReLU(True),
            nn.Conv3d(base, base*2, 4, 2, 1), nn.BatchNorm3d(base*2), nn.ReLU(True),
            nn.Conv3d(base*2, base*4, 4, 2, 1), nn.BatchNorm3d(base*4), nn.ReLU(True),

            # 反卷积解码
            nn.ConvTranspose3d(base*4, base*2, 4, 2, 1), nn.BatchNorm3d(base*2), nn.ReLU(True),
            nn.ConvTranspose3d(base*2, base, 4, 2, 1), nn.BatchNorm3d(base), nn.ReLU(True),
            nn.ConvTranspose3d(base, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator3D(nn.Module):
    def __init__(self, in_ch=5, base=32):
        super().__init__()
        self.net = nn.Sequential(

            # 只对 X/Y 下采样，Z 不变
            nn.Conv3d(in_ch, base, (4,4,1), (2,2,1), (1,1,0)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(base, base*2, (4,4,1), (2,2,1), (1,1,0)),
            nn.BatchNorm3d(base*2),
            nn.LeakyReLU(0.2),

            nn.Conv3d(base*2, base*4, (4,4,1), (2,2,1), (1,1,0)),
            nn.BatchNorm3d(base*4),
            nn.LeakyReLU(0.2),

            # 最终输出为 [B, 1, 3, 3, 2700]
            nn.Conv3d(base*4, 1, (3,3,1), (1,1,1), (0,0,0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        # 判别器输出一维向量（B个预测）
        return torch.mean(out, dim=(1,2,3,4))   # shape: [B]



# ------------------------------
# 可视化 voxel
# ------------------------------
def visualize_voxel(voxel_5ch, channel=1, opacity=0.3):
    nx, ny, nz, nch = voxel_5ch.shape
    voxel_values = voxel_5ch[..., channel]

    # mask
    mask = voxel_values > 0.05
    xv, yv, zv = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    idxs = np.stack([xv, yv, zv], axis=-1).reshape(-1,3)
    mask_flat = mask.ravel()
    selected_idx = idxs[mask_flat]

    if selected_idx.size == 0:
        print("No voxel to display")
        return

    voxel_centers = selected_idx.astype(np.float32)
    cube_size = 1.0

    pts_list = []
    cells = []
    cell_types = []
    for i, c in enumerate(voxel_centers):
        x0,y0,z0 = c
        pts = np.array([
            [x0,y0,z0],
            [x0+cube_size,y0,z0],
            [x0+cube_size,y0+cube_size,z0],
            [x0,y0+cube_size,z0],
            [x0,y0,z0+cube_size],
            [x0+cube_size,y0,z0+cube_size],
            [x0+cube_size,y0+cube_size,z0+cube_size],
            [x0,y0+cube_size,z0+cube_size]
        ], dtype=np.float32)
        pts_list.append(pts)
        start_idx = i*8
        cells.append(np.hstack([[8], np.arange(start_idx,start_idx+8)]))
        cell_types.append(pv.CellType.HEXAHEDRON)

    pts_array = np.vstack(pts_list)
    cells_array = np.hstack(cells).astype(np.int64)
    cell_types = np.array(cell_types)
    grid = pv.UnstructuredGrid(cells_array, cell_types, pts_array)
    grid.cell_data["values"] = voxel_values.ravel()[mask_flat]
    pl = pv.Plotter()
    pl.add_mesh(grid, scalars="values", opacity=opacity, cmap='viridis')
    pl.show()

# ------------------------------
# 主训练
# ------------------------------
def train_3dgan(seismic_dir, voxel_in, device='cuda', epochs=5, batch_size=2, lr=2e-4):
    dataset = SeismicVoxelDataset(seismic_dir, voxel_in)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator3D().to(device)
    D = Discriminator3D().to(device)

    criterion = nn.BCELoss()
    mse_loss = nn.MSELoss()   # 可选：融合重建质量
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for x, y_real in loader:
            x = x.to(device)
            y_real = y_real.to(device)

            bsize = x.size(0)
            real_label = torch.ones(bsize, device=device)
            fake_label = torch.zeros(bsize, device=device)

            # -----------------------------
            # 1. 训练 Discriminator
            # -----------------------------
            D.zero_grad()
            # 判别真实
            out_real = D(y_real)
            loss_D_real = criterion(out_real, real_label)

            # 判别生成
            y_fake = G(x)
            y_fake = match_target_z(y_fake, target_z=y_real.shape[-1])

            out_fake = D(y_fake.detach())
            loss_D_fake = criterion(out_fake, fake_label)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            opt_D.step()

            # -----------------------------
            # 2. 训练 Generator
            # -----------------------------
            G.zero_grad()
            out_fake2 = D(y_fake)
            loss_G_adv = criterion(out_fake2, real_label)
            loss_G_recon = mse_loss(y_fake, y_real) * 10.0  # 让 voxel 更贴近真实

            loss_G = loss_G_adv + loss_G_recon
            loss_G.backward()
            opt_G.step()

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Loss_D={loss_D.item():.4f}, "
              f"Loss_G={loss_G.item():.4f}")

    return G, D

# ------------------------------
# 推理示例
# ------------------------------
def inference(generator, seismic_block):
    generator.eval()
    with torch.no_grad():
        x = torch.from_numpy(seismic_block.astype(np.float32)).permute(3,0,1,2)[None,...]  # [1,1,24,24,Z]
        y_pred = generator(x)  # [1,5,24,24,Z]
        y_pred_np = y_pred.squeeze(0).permute(1,2,3,0).cpu().numpy()  # [24,24,Z,5]
    return y_pred_np

# ------------------------------
# 示例运行
# ------------------------------
if __name__=="__main__":
    seismic_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容三/3D_GAN输入/seismic"
    voxel_in = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容三/训练用数据/体素表示"
    out_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容三/3D_GAN输出"
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "模型")
    os.makedirs(model_dir, exist_ok=True)
    voxel_out = os.path.join(out_dir, "体素文件")
    os.makedirs(voxel_out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # 训练及保存
    # -------------------------------
    # G, D = train_3dgan(seismic_dir, voxel_in, device=device, epochs=3, batch_size=2)
    #
    # torch.save(G.state_dict(), os.path.join(model_dir, "G_3dgan.pth"))
    # torch.save(D.state_dict(), os.path.join(model_dir, "D_3dgan.pth"))
    # print("✓ 已保存 3D-GAN 生成器和判别器")

    # -------------------------------
    # 使用已训练好的生成器
    # -------------------------------
    G = Generator3D().to(device)
    G.load_state_dict(torch.load(os.path.join(out_dir, "模型", "G_3dgan.pth"), map_location=device))
    G.eval()
    print(f"✓ 已加载训练好的生成器")

    # -------------------------------
    # 推理
    # -------------------------------


    # 保存推理生成的体素文件
    # for i in range(85):
    for i in range(1):
        # for j in range(61):
        for j in range(2):
            sample_seismic = np.load(f"{seismic_dir}/sample_X{i}_Y{j}.npy")
            voxel_pred = inference(G, sample_seismic)

            voxel_file = os.path.join(voxel_out, f"voxel_pred_X{i}_Y{j}.npy")
            np.save(voxel_file, voxel_pred)
            print(f"✓ 推理生成的体素文件已保存: {voxel_file}")

            # 可视化
            visualize_voxel(voxel_pred, channel=4, opacity=0.3)
