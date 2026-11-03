import torch
import math
import numpy as np
from torchvision.transforms.functional import rotate, InterpolationMode
import torch.nn.functional as F
# def my_rotate(img, interpolation=InterpolationMode.BILINEAR, angle=None):
#     """
#     自定义旋转函数，用于旋转 PSF。

#     参数:
#         img: 输入的图像或 PSF，形状为 (C, H, W)。
#         interpolation: 插值模式，默认为双线性插值。
#         angle: 旋转角度（度）。

#     返回:
#         旋转后的图像或 PSF。
#     """
#     if angle is None:
#         return img

#     # 将角度转换为弧度
#     angle_rad = angle * math.pi / 180

#     # 构造旋转矩阵
#     n, c, h, w = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
#     theta = torch.zeros(n, 2, 3, device=img.device, dtype=img.dtype)
#     theta[:, 0, 0] = torch.cos(angle_rad)
#     theta[:, 0, 1] = -torch.sin(angle_rad)
#     theta[:, 1, 0] = torch.sin(angle_rad)
#     theta[:, 1, 1] = torch.cos(angle_rad)

#     # 构造网格
#     grid = torch.nn.functional.affine_grid(theta, img.size(), align_corners=False)

#     # 应用旋转
#     img_rotated = torch.nn.functional.grid_sample(img, grid, mode=interpolation.value, padding_mode="zeros", align_corners=False)
#     return img_rotated

def rotate_image(img, angle, interpolation='bilinear'):
    """
    自定义旋转函数，用于旋转图像。

    参数:
        img: 输入的图像 tensor，形状为 [1, 3, H, W]。
        angle: 旋转角度（度）float。
        interpolation: 插值模式，默认为双线性插值。

    返回:
        旋转后的图像 tensor，形状为 [1, 3, H, W]。
    """

    device = img.device
    # 如果 angle 是 torch.Tensor，将其转换为 Python 的 float
    if isinstance(angle, torch.Tensor):
        angle = angle.item()  # 提取标量值
    # 将角度转换为弧度
    angle_rad = math.radians(angle)

    # 获取图像的高度和宽度
    _, _, H, W = img.shape

    # 创建旋转矩阵
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    # 旋转矩阵
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0]
    ], dtype=torch.float32,device=device).unsqueeze(0)  # 添加 batch 维度

    # 创建仿射变换的网格
    grid = F.affine_grid(rotation_matrix, img.size(), align_corners=False).to(device)

    # 使用 grid_sample 进行旋转
    if interpolation == 'bilinear':
        mode = 'bilinear'
    elif interpolation == 'nearest':
        mode = 'nearest'
    else:
        raise ValueError("Unsupported interpolation mode. Use 'bilinear' or 'nearest'.")

    rotated_img = F.grid_sample(img, grid, mode=mode, padding_mode='zeros', align_corners=False)

    return rotated_img

def get_patch_psf(psf, view_pos, patch_top_left, img_size, patch_length, device):
    """
    根据 patch 的左上角坐标，生成该 patch 对应的 PSF。

    参数:
        psf: 输入的 PSF 数据，形状为 (64, 3, 41, 41)。
        view_pos: 视场位置，用于插值，例如 [0.0, 0.25, 0.5, 0.75, 1.0]。
        patch_top_left: patch 的左上角坐标 (x_min, y_min)。
        img_size: 图像的尺寸 (height, width)。
        patch_length: 每个 patch 的长度。
        device: 计算设备（如 'cpu' 或 'cuda'）。

    返回:
        patch_psf: 裁剪 patch 对应的 PSF,形状为 (3, 41, 41)。
    """
    h, w = img_size[0], img_size[1]
    h_num = round(h / patch_length)  # 高度方向上的 patch 数量
    w_num = round(w / patch_length)  # 宽度方向上的 patch 数量

    # 计算 patch 的中心位置（像素单位）
    x_min, y_min = patch_top_left
    x_center = x_min + patch_length / 2  # 像素单位
    y_center = y_min + patch_length / 2  # 像素单位
    # print('x_center:',x_center,'y_center:',y_center)
    # 将 patch 中心坐标转换为 patch 数量单位
    x_center_patch = x_center / patch_length
    y_center_patch = y_center / patch_length
    # print('x_center_patch:',x_center_patch,'y_center_patch:',y_center_patch)
    # 计算 patch 中心到图像中心的距离（patch 数量单位）
    d_loc = np.sqrt((x_center_patch - w_num / 2) ** 2 + (y_center_patch - h_num / 2) ** 2)

    # 计算最大距离（patch 数量单位）
    d_max = np.sqrt(w_num ** 2 + h_num ** 2) / 2

    # 计算 patch 中心相对于图像中心的角度
    standard_vector = torch.Tensor([0, 1]).to(device)
    yloc = y_center_patch - h_num / 2
    xloc = x_center_patch - w_num / 2
    # print('xloc:',xloc,'yloc:',yloc)
    location_vector = torch.Tensor([xloc, yloc]).to(device)
    norm_l = torch.norm(location_vector)
    cos_angle = torch.dot(location_vector, standard_vector) / norm_l
    rotate_theta = torch.arccos(cos_angle) / math.pi * 180
    if xloc > 0:
        rotate_theta = -rotate_theta
    # rotate_theta = rotate_theta + 180
    rotate_theta = 180 - rotate_theta
    print('rotate theta',rotate_theta)
    # 插值得到 patch 的 PSF
    d_patch = d_loc / d_max  # 归一化距离
    #判断是否存在nan
    if torch.isnan(psf).sum() > 0:
        print('nan exists in psf')
    #view_pos = np.linspace(0, 1, 64)
    eps = 1e-6
    index = (torch.from_numpy(view_pos).float().to(device) - d_patch + eps) ** (-2)
    index = index / index.sum()
    # print('index',index)
    psf_used = (psf * index.view(-1, 1, 1, 1)).sum(dim=0)  # 插值得到 PSF

    #判断是否存在nan
    if torch.isnan(psf_used).sum() > 0:
        print('nan exists in psf_used')

    # 旋转 PSF
    psf_used = rotate_image(psf_used.unsqueeze(0), angle=rotate_theta)
    psf_used = psf_used.squeeze(0)
    #判断是否存在nan
    if torch.isnan(psf_used).sum() > 0:
        print('nan exists in rotate psf_used')
    # # 归一化 PSF
    # psf_used = psf_used / psf_used.sum(dim=2).sum(dim=1).unsqueeze(1).unsqueeze(2)

    # return psf_used,non_rotate_psf_used
    return psf_used

import torch
import numpy as np
import math

def get_patch_psf_batch(psf, view_pos, patch_top_left_list, img_size, patch_length, device):
    """
    根据一系列 patch 的左上角坐标，并行生成对应的 PSF。

    参数:
        psf: 输入的 PSF 数据，形状为 (64, 3, 41, 41)。
        view_pos: 视场位置，用于插值，例如 [0.0, 0.25, 0.5, 0.75, 1.0]。
        patch_top_left_list: 一系列 patch 的左上角坐标，形状为 (N, 2)，其中 N 是 patch 的数量。
        img_size: 图像的尺寸 (height, width)。
        patch_length: 每个 patch 的长度。
        device: 计算设备（如 'cpu' 或 'cuda'）。

    返回:
        patch_psf_batch: 裁剪 patch 对应的 PSF，形状为 (N, 3, 41, 41)。
    """
    h, w = img_size[0], img_size[1]
    h_num = round(h / patch_length)  # 高度方向上的 patch 数量
    w_num = round(w / patch_length)  # 宽度方向上的 patch 数量

    # 将 patch_top_left_list 转换为 Tensor
    patch_top_left_list = torch.tensor(patch_top_left_list, dtype=torch.float32, device=device)

    # 计算 patch 的中心位置（像素单位）
    x_min = patch_top_left_list[:, 0]
    y_min = patch_top_left_list[:, 1]
    x_center = x_min + patch_length / 2  # 像素单位
    y_center = y_min + patch_length / 2  # 像素单位

    # 将 patch 中心坐标转换为 patch 数量单位
    x_center_patch = x_center / patch_length
    y_center_patch = y_center / patch_length

    # 计算 patch 中心到图像中心的距离（patch 数量单位）
    d_loc = torch.sqrt((x_center_patch - w_num / 2) ** 2 + (y_center_patch - h_num / 2) ** 2)

    # 计算最大距离（patch 数量单位）
    d_max = torch.sqrt(torch.tensor(w_num ** 2 + h_num ** 2, device=device)) / 2

    # 计算 patch 中心相对于图像中心的角度
    standard_vector = torch.tensor([0, 1], dtype=torch.float32, device=device)
    yloc = y_center_patch - h_num / 2
    xloc = x_center_patch - w_num / 2
    location_vector = torch.stack([xloc, yloc], dim=1)  # (N, 2)
    norm_l = torch.norm(location_vector, dim=1, keepdim=True)
    cos_angle = torch.sum(location_vector * standard_vector, dim=1) / norm_l.squeeze()
    rotate_theta = torch.arccos(cos_angle) / math.pi * 180
    rotate_theta[xloc > 0] = -rotate_theta[xloc > 0]
    rotate_theta = 180 - rotate_theta

    # 插值得到 patch 的 PSF
    d_patch = d_loc / d_max  # 归一化距离
    # print('d_patch:',d_patch.shape)
    eps = 1e-6
    index = (torch.from_numpy(view_pos).float().to(device) - d_patch.unsqueeze(1) + eps) ** (-2)  # (N, 64)
    index = index / index.sum(dim=1, keepdim=True)  # 归一化
    # print('index:',index.shape)
    psf_used = torch.sum(psf.unsqueeze(0) * index.view(-1, 64, 1, 1, 1), dim=1)  # (N, 3, 41, 41)
    # print('psf_used:',psf.unsqueeze(0).shape,'index',index.view(-1, 64, 1, 1, 1).shape)
    # 旋转 PSF
    patch_psf_batch = []
    for i in range(psf_used.shape[0]):
        rotated_psf = rotate_image(psf_used[i].unsqueeze(0), angle=rotate_theta[i])
        patch_psf_batch.append(rotated_psf.squeeze(0))
    patch_psf_batch = torch.stack(patch_psf_batch, dim=0)  # (N, 3, 41, 41)

    #统一尺寸
    patch_psf_batch = F.interpolate(patch_psf_batch, (61, 61), mode='bilinear', align_corners=False)
    return patch_psf_batch

#主函数
if __name__ == '__main__':

    # import torch
    # from torchvision.utils import save_image
    # #测试旋转函数
    # psf = torch.load('kernels/s3/piece1_fov20_fnum5.0_aper1_imgr14.3_ttl32.9_bfl22.0_aspFalse_rms0.0773_idx0.pth')
    # psf = psf[30] #[3,61,61]
    # # print('min:',psf.min(),'max:',psf.max())
    # #逐第一个通道归一化
    # # min_val = psf.min(dim=(1,2),keepdim=True).values
    # # max_val = psf.max(dim=(1,2),keepdim=True).values
    # norm_psf = (psf)/(psf.max())
    # print('min:',norm_psf.min(),'max:',norm_psf.max())
    # save_image(norm_psf,'psf.png')
    # #save psf img
    # import cv2
    # import numpy as np
    # psf_rotated = rotate_image(norm_psf.unsqueeze(0),angle=213.69)
    # psf_rotated = psf_rotated.squeeze(0)
    # #norm
    # norm_psf_rotated = (psf_rotated)/(psf_rotated.max())
    # print('min:',norm_psf_rotated.min(),'max:',norm_psf_rotated.max())
    # save_image(norm_psf_rotated,'psf_rotated.png')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # psf = torch.load('kernels/p2/piece2_fov30_fnum4.2_aper2_imgr14_ttl76.9_bfl7.7_aspFalse_rms0.0484_idx0.pth').to(device)
    # view_pos = np.linspace(0, 1, 64)
    # img_size = (1280, 1920)
    # patch_length = 128
    # patch_coords = (1800, 1100)
    # # patch_coords = (0, 0)
    # patch_psf, non_rorate_psf = get_patch_psf(psf, view_pos, patch_coords, img_size, patch_length, device)
    # print(patch_psf.shape)
    # #save
    # import cv2
    # import numpy as np
    # patch_psf = patch_psf.cpu().numpy()
    # #逐通道归一化
    # patch_psf = patch_psf/patch_psf.max()
    # #save psf img
    # patch_psf = np.transpose(patch_psf,(1,2,0))
    # patch_psf = cv2.cvtColor(patch_psf, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('patch_psf.png', np.clip(255*patch_psf, 0, 255).astype(np.uint8))

    # non_rorate_psf = non_rorate_psf.cpu().numpy()
    # #逐通道归一化
    # non_rorate_psf = non_rorate_psf/non_rorate_psf.max()
    # #save psf img
    # non_rorate_psf = np.transpose(non_rorate_psf,(1,2,0))
    # non_rorate_psf = cv2.cvtColor(non_rorate_psf, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('non_rorate_psf.png', np.clip(255*non_rorate_psf, 0, 255).astype(np.uint8))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 示例输入
    psf = torch.randn(64, 3, 41, 41).to(device)  # 假设 PSF 数据已经加载
    view_pos = np.linspace(0, 1, 64)  # 视场位置
    patch_top_left_list = np.array([[0, 0], [32, 32], [64, 64]])  # 示例坐标
    img_size = (128, 128)  # 图像尺寸
    patch_length = 32  # patch 长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 调用函数
    patch_psf_batch = get_patch_psf_batch(psf, view_pos, patch_top_left_list, img_size, patch_length, device)
    print(patch_psf_batch.shape)  # 输出: torch.Size([3, 3, 41, 41])