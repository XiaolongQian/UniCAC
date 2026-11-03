# import numpy as np
# import torch
# from scipy.ndimage import rotate, map_coordinates

# # def calculate_energy_radius(psf_tensor):
# #     """
# #     计算每个PSF在H和W方向上的能量分布半径。

# #     参数:
# #         psf_tensor (torch.Tensor): 形状为 [N, H, W, C] 的PSF张量。

# #     返回:
# #         energy_radius (torch.Tensor): 形状为 [N, 2] 的能量分布半径张量。
# #     """
# #     # 计算每个PSF在H和W方向上的标准差
# #     h_radius = torch.std(psf_tensor, dim=1)  # 在H方向上的标准差
# #     w_radius = torch.std(psf_tensor, dim=2)  # 在W方向上的标准差

# #     # 取平均值得到每个PSF的H和W方向的能量分布半径
# #     h_radius = h_radius.mean(dim=(1, 2))  # [N]
# #     w_radius = w_radius.mean(dim=(1, 2))  # [N]

# #     # 组合成 [N, 2] 的张量
# #     energy_radius = torch.stack([h_radius, w_radius], dim=1)  # [N, 2]

# #     return energy_radius

def calculate_fwhm(psf_tensor):
    """
    计算每个PSF在H和W方向上的半高全宽（FWHM）。

    参数:
        psf_tensor (torch.Tensor): 形状为 [N, H, W, C] 的PSF张量。

    返回:
        fwhm (torch.Tensor): 形状为 [N, 2] 的张量，表示每个PSF在H和W方向上的FWHM。
    """
    #[N,C,H,W] _> [N,H,W,C]
    psf_tensor = psf_tensor.permute(0, 2, 3, 1)
    num_psf, h, w, _ = psf_tensor.shape
    fwhm = torch.zeros(num_psf, 2)  # 存储每个PSF的H和W方向的FWHM

    for i in range(num_psf):
        # 取当前PSF的强度（可以对通道取平均）
        psf = psf_tensor[i].mean(dim=-1)  # 形状为 [H, W]

        # 计算H方向上的FWHM
        h_profile = psf.max(dim=1).values  # 在H方向上取最大值
        h_fwhm = compute_fwhm_1d(h_profile)  # 计算H方向的FWHM

        # 计算W方向上的FWHM
        w_profile = psf.max(dim=0).values  # 在W方向上取最大值
        w_fwhm = compute_fwhm_1d(w_profile)  # 计算W方向的FWHM

        # 存储结果
        fwhm[i, 0] = h_fwhm
        fwhm[i, 1] = w_fwhm

    return fwhm

def compute_fwhm_1d(profile):
    """
    计算一维数据的半高全宽（FWHM）。

    参数:
        profile (torch.Tensor): 一维数据，形状为 [L]。

    返回:
        fwhm (float): 半高全宽。
    """
    max_value = profile.max()  # 最大值
    half_max = max_value / 2  # 半高

    # 找到大于半高的区域
    above_half_max = profile > half_max

    # 找到区域的起始和结束位置
    indices = torch.where(above_half_max)[0]
    if len(indices) == 0:
        return 0.0  # 如果没有大于半高的区域，返回0

    start = indices[0]
    end = indices[-1]

    # 计算FWHM
    fwhm = end - start + 1  # 加1是因为包括起始和结束点
    return fwhm

# # def create_energy_distribution_map(psf_tensor, image_size):
# #     """
# #     根据PSF张量和图像尺寸生成能量分布图。

# #     参数:
# #         psf_tensor (torch.Tensor): 形状为 [N, H, W, C] 的PSF张量。
# #         image_size (tuple): 目标图像的尺寸 (H, W)。

# #     返回:
# #         energy_distribution_map (torch.Tensor): 形状为 [H, W, 2] 的能量分布图。
# #     """
# #     # 计算每个PSF的能量分布半径
# #     # energy_radius = calculate_energy_radius(psf_tensor)
# #     energy_radius = calculate_fwhm(psf_tensor)

# #     h, w = image_size
# #     num_psf = energy_radius.shape[0]
# #     psf_h, psf_w = psf_tensor.shape[1:3]  # 获取PSF的H和W尺寸

# #     # 创建一个空的能量分布图
# #     energy_map = torch.zeros(h, w, 2)

# #     # 计算图像的中心点
# #     center = torch.tensor([h // 2, w // 2])

# #     # 遍历每个PSF的能量分布半径
# #     for i in range(num_psf):
# #         # 获取当前PSF的H和W方向的能量分布半径
# #         h_radius, w_radius = energy_radius[i]

# #         # 创建一个小的能量分布图，尺寸与PSF相同
# #         small_energy_map = torch.zeros(psf_h, psf_w, 2)
# #         small_energy_map[..., 0] = h_radius
# #         small_energy_map[..., 1] = w_radius

# #         # 旋转小能量分布图以匹配图像的半对角线
# #         angle = np.arctan2(h, w) * 180 / np.pi
# #         rotated_energy_map = rotate(small_energy_map.numpy(), angle, reshape=False)

# #         # 生成插值坐标
# #         x = np.linspace(0, psf_w - 1, w)  # PSF的W方向坐标
# #         y = np.linspace(0, psf_h - 1, h)  # PSF的H方向坐标
# #         xv, yv = np.meshgrid(x, y)

# #         # 将坐标调整为 (2, h * w) 的形状
# #         coordinates = np.stack([yv.ravel(), xv.ravel()], axis=0)  # (2, h * w)

# #         # 对旋转后的能量分布图的每个通道分别进行插值
# #         interpolated_energy_map = np.zeros((h, w, 2))
# #         for channel in range(2):
# #             interpolated_energy_map[..., channel] = map_coordinates(
# #                 rotated_energy_map[..., channel], coordinates, order=1
# #             ).reshape(h, w)

# #         # 将插值后的能量分布图添加到总能量分布图中
# #         energy_map += torch.tensor(interpolated_energy_map)

# #     return energy_map
# def create_energy_distribution_map(psf_tensor, image_size):
#     """
#     根据PSF张量和图像尺寸生成能量分布图。

#     参数:
#         psf_tensor (torch.Tensor): 形状为 [N, H, W, C] 的PSF张量。
#         image_size (tuple): 目标图像的尺寸 (H, W)。
#         fov_values (torch.Tensor): 每个PSF对应的视场角（FOV）值，形状为 [N]。

#     返回:
#         energy_map (torch.Tensor): 形状为 [H, W, 2] 的能量分布图。
#     """
#     fov_values = torch.linspace(0, 1, psf_tensor.shape[0])  # 生成视场角（FOV）值
#     h, w = image_size
#     num_psf = psf_tensor.shape[0]

#     # 计算每个PSF的FWHM
#     fwhm = calculate_fwhm(psf_tensor)  # 形状为 [N, 2]

#     # 创建一个空的能量分布图
#     energy_map = torch.zeros(h, w, 2)

#     # 计算图像的中心点
#     center = torch.tensor([h // 2, w // 2])

#     # 遍历图像的每个像素
#     for y in range(h):
#         for x in range(w):
#             # 计算当前像素到图像中心的距离（归一化到 [0, 1]）
#             distance = torch.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
#             normalized_distance = distance / torch.sqrt(center[0] ** 2 + center[1] ** 2)

#             # 根据距离选择合适的FOV值
#             fov_index = torch.argmin(torch.abs(fov_values - normalized_distance))
#             selected_fwhm = fwhm[fov_index]  # 形状为 [2]

#             # 将FWHM值填充到能量分布图中
#             energy_map[y, x, 0] = selected_fwhm[0]  # H方向的FWHM
#             energy_map[y, x, 1] = selected_fwhm[1]  # W方向的FWHM

#     return energy_map

# # 示例调用
# psf_tensor = torch.randn(64, 51, 51, 3)  # 示例PSF张量，H=40, W=60
# image_size = (1280, 1920)  # 目标图像尺寸

# # 生成能量分布图
# energy_distribution_map = create_energy_distribution_map(psf_tensor, image_size)
# print(energy_distribution_map.shape)  # torch.Size([1280, 1920, 2])



import torch
import numpy as np
def calculate_fov_map(image_size, fov_values):
    """
    计算整张图的 FOV 信息（向量化实现）。

    参数:
        image_size (tuple): 目标图像的尺寸 (H, W)。
        fov_values (torch.Tensor): 每个PSF对应的视场角（FOV）值，形状为 [N]。

    返回:
        fov_map (torch.Tensor): 形状为 [H, W] 的 FOV 信息图。
    """
    h, w = image_size
    fov_values = fov_values.numpy()  # 将 fov_values 转换为 NumPy 数组以便与 NumPy 操作兼容

    # 生成图像的坐标网格
    h_range = np.arange(0, h, 1)
    w_range = np.arange(0, w, 1)
    img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)

    # 将坐标归一化到 [-1, 1]
    img_fld_h = ((img_fld_h - (h - 1) / 2) / ((h - 1) / 2)).astype(np.float32)
    img_fld_w = ((img_fld_w - (w - 1) / 2) / ((w - 1) / 2)).astype(np.float32)

    # 计算每个像素到图像中心的距离（归一化到 [0, 1]）
    distance = np.sqrt(img_fld_h ** 2 + img_fld_w ** 2)  # 形状为 [H, W]
    normalized_distance = distance / np.sqrt(2)  # 最大距离为 sqrt(2)，归一化到 [0, 1]

    # 根据距离选择合适的 FOV 值
    fov_map = np.argmin(np.abs(fov_values[:, np.newaxis, np.newaxis] - normalized_distance), axis=0)  # 形状为 [H, W]

    # 将结果转换为 PyTorch 张量
    fov_map = torch.from_numpy(fov_map)

    return fov_map

def random_crop_fov_map(fov_map, crop_size=(256, 256)):
    """
    从 FOV 信息图中随机裁剪一个区域。

    参数:
        fov_map (torch.Tensor): 形状为 [H, W] 的 FOV 信息图。
        crop_size (tuple): 裁剪区域的尺寸 (H, W)，默认为 (256, 256)。

    返回:
        fov_map_crop (torch.Tensor): 形状为 [crop_H, crop_W] 的裁剪区域 FOV 信息图。
    """
    h, w = fov_map.shape
    crop_h, crop_w = crop_size

    # 随机生成裁剪区域的起始坐标
    start_y = torch.randint(0, h - crop_h + 1, (1,)).item()
    start_x = torch.randint(0, w - crop_w + 1, (1,)).item()

    # 裁剪 FOV 信息图
    fov_map_crop = fov_map[start_y:start_y + crop_h, start_x:start_x + crop_w]

    return fov_map_crop

def create_energy_map_from_fov(fov_map_crop, fwhm):
    """
    根据裁剪区域的 FOV 信息图生成能量分布图。

    参数:
        fov_map_crop (torch.Tensor): 形状为 [crop_H, crop_W] 的裁剪区域 FOV 信息图。
        fwhm (torch.Tensor): 每个PSF的 FWHM 值，形状为 [N, 2]。

    返回:
        energy_map_crop (torch.Tensor): 形状为 [crop_H, crop_W, 2] 的能量分布图。
    """
    crop_h, crop_w = fov_map_crop.shape
    energy_map_crop = torch.zeros(crop_h, crop_w, 2)

    # 遍历裁剪区域的每个像素
    for y in range(crop_h):
        for x in range(crop_w):
            # 获取当前像素的 FOV 索引
            fov_index = int(fov_map_crop[y, x].item())

            # 从 fwhm 中提取对应的 FWHM 值
            selected_fwhm = fwhm[fov_index]  # 形状为 [2]

            # 将 FWHM 值填充到裁剪区域的能量分布图中
            energy_map_crop[y, x, 0] = selected_fwhm[0]  # H方向的FWHM
            energy_map_crop[y, x, 1] = selected_fwhm[1]  # W方向的FWHM

    return energy_map_crop

# # 示例调用
# psf_tensor = torch.randn(64, 51, 51, 3)  # 示例PSF张量，H=40, W=60
# image_size = (1280, 1920)  # 目标图像尺寸
# fov_values = torch.linspace(0, 1, 64)  # 假设每个PSF对应一个FOV值

# # 计算整张图的 FOV 信息
# fov_map = calculate_fov_map(image_size, fov_values)
# print(fov_map.shape)  # torch.Size([1280, 1920])
# #tensor -> numpy
# fov_map_np = fov_map.numpy()
# #save fov_map_np
# np.save('/root/autodl-tmp/home/qxl/dataset/fov_map_1280_1920.npy',fov_map_np)

# # 随机裁剪一个区域
# fov_map_crop = random_crop_fov_map(fov_map, crop_size=(256, 256))

# # 计算每个PSF的FWHM
# fwhm = calculate_fwhm(psf_tensor)  # 形状为 [N, 2]

# # 生成裁剪区域的能量分布图
# energy_map_crop = create_energy_map_from_fov(fov_map_crop, fwhm)
# print(energy_map_crop.shape)  # torch.Size([256, 256, 2])