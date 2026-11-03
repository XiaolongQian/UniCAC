from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment_fov, paired_random_crop_fov
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import numpy as np
import torch
from basicsr.data.psf_energy import calculate_fwhm,create_energy_map_from_fov
import os
import glob
@DATASET_REGISTRY.register()
class PairedImage_PSF_FWHM_Dataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImage_PSF_FWHM_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        self.fov_map = np.load(opt['fov_map_path'])
        print('fov_map',self.fov_map.shape)
        self.psf_path = opt['psf_path']
        # self.txt_path = sorted(glob.glob(os.path.join(opt['txt_path'], '*.txt')))
        self.txt_path = opt['txt_path']
        # print('txt_path',self.txt_path)
    def __getitem__(self, index):
        # print('index',index)
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        # print('img_lq',img_lq.shape)
        # print('gt_path',gt_path,'lq_path',lq_path)
        fov = self.fov_map
        #(1280,1920) -> (1280,1920,1)
        fov = fov[..., None]
        #rgb name
        gt_name = os.path.basename(gt_path)
        gt_name = os.path.splitext(gt_name)[0]

        #read index txt file
        txt_path = os.path.join(self.txt_path, gt_name + '.txt')
        # print('index_txt',txt_path)

        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read().rstrip('\n')  # 读取整个文件内容
            # print('psf',content)
        #根据self.psf_path和content路径连在一起读取pth文件
        psf_path = os.path.join(self.psf_path, content)
        psf = torch.load(psf_path, map_location='cpu')
        # psf = torch.randn(64, 51, 51, 3)  # 示例PSF张量，H=40, W=60
        fwhm = calculate_fwhm(psf)  # 形状为 [N, 2]
        # print('fwhm',fwhm.shape)


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, fov = paired_random_crop_fov(img_gt, img_lq, fov,gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq, fov = augment_fov([img_gt, img_lq,fov], self.opt['use_hflip'], self.opt['use_rot'])
        #fov numpy (h,w,1) ->  tensor [1,h,w]
        fov = torch.from_numpy(fov.copy().transpose(2, 0, 1)).float()
        #[1,h,w] -> [h,w]
        fov = fov.squeeze(0)
        energy_map_crop = create_energy_map_from_fov(fov, fwhm)
        #按通道归一化
        energy_map_crop = energy_map_crop / energy_map_crop.max()
        #[h,w,2] -> [2,h,w]
        energy_map_crop = energy_map_crop.permute(2, 0, 1)
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # fov = torch.from_numpy(fov.copy().transpose(2, 0, 1)).float()
        #img_lq fov concat
        # img_lq = torch.cat([img_lq, energy_map_crop], 0)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        #print shape
        # print('img_lq',img_lq.shape,'img_gt',img_gt.shape)
        return {'lq': img_lq, 'gt': img_gt, 'psf':energy_map_crop, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImage_PSF_FWHM_test_Dataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImage_PSF_FWHM_test_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        self.fov_map = np.load(opt['fov_map_path'])
        print('fov_map',self.fov_map.shape)
        self.psf_path = opt['psf_path']
        # self.txt_path = sorted(glob.glob(os.path.join(opt['txt_path'], '*.txt')))
        # self.txt_path = opt['txt_path']
        # print('txt_path',self.txt_path)
    def __getitem__(self, index):
        # print('index',index)
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        # print('img_lq',img_lq.shape)
        # print('gt_path',gt_path,'lq_path',lq_path)
        fov = self.fov_map
        #(1280,1920) -> (1280,1920,1)
        fov = fov[..., None]

        lens_name = self.gt_folder.split('/')[-2]
        print('lens_name',lens_name)
        psf_path = self.psf_path + '/' + lens_name + '.npy'
        psf = torch.from_numpy(np.load(psf_path)) #(64,3,h,w)
        # #rgb name
        # gt_name = os.path.basename(gt_path)
        # gt_name = os.path.splitext(gt_name)[0]

        # #read index txt file
        # txt_path = os.path.join(self.txt_path, gt_name + '.txt')
        # # print('index_txt',txt_path)

        # with open(txt_path, 'r', encoding='utf-8') as file:
        #     content = file.read().rstrip('\n')  # 读取整个文件内容
        #     # print('psf',content)
        # #根据self.psf_path和content路径连在一起读取pth文件
        # psf_path = os.path.join(self.psf_path, content)
        # psf = torch.load(psf_path, map_location='cpu')
        # psf = torch.randn(64, 51, 51, 3)  # 示例PSF张量，H=40, W=60
        fwhm = calculate_fwhm(psf)  # 形状为 [N, 2]
        # print('fwhm',fwhm.shape)


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, fov = paired_random_crop_fov(img_gt, img_lq, fov,gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq, fov = augment_fov([img_gt, img_lq,fov], self.opt['use_hflip'], self.opt['use_rot'])
        #fov numpy (h,w,1) ->  tensor [1,h,w]
        fov = torch.from_numpy(fov.copy().transpose(2, 0, 1)).float()
        #[1,h,w] -> [h,w]
        fov = fov.squeeze(0)
        energy_map_crop = create_energy_map_from_fov(fov, fwhm)
        #按通道归一化
        energy_map_crop = energy_map_crop / energy_map_crop.max()
        #[h,w,2] -> [2,h,w]
        energy_map_crop = energy_map_crop.permute(2, 0, 1)
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # fov = torch.from_numpy(fov.copy().transpose(2, 0, 1)).float()
        #img_lq fov concat
        # img_lq = torch.cat([img_lq, energy_map_crop], 0)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        #print shape
        # print('img_lq',img_lq.shape,'img_gt',img_gt.shape)
        return {'lq': img_lq, 'gt': img_gt, 'psf':energy_map_crop, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)



import torch
import torch.nn as nn
@DATASET_REGISTRY.register()
class PART_PSF_FWHM_Dataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PART_PSF_FWHM_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        self.fov_map = np.load(opt['fov_map_path'])
        print('fov_map',self.fov_map.shape)
        self.psf_path = opt['psf_path']
        # self.txt_path = sorted(glob.glob(os.path.join(opt['txt_path'], '*.txt')))
        self.txt_path = opt['txt_path']
        # print('txt_path',self.txt_path)

        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=4)

    def __getitem__(self, index):
        # print('index',index)
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        # print('img_lq',img_lq.shape)
        # print('gt_path',gt_path,'lq_path',lq_path)
        fov = self.fov_map
        #(1280,1920) -> (1280,1920,1)
        fov = fov[..., None]
        #rgb name
        gt_name = os.path.basename(gt_path)
        gt_name = os.path.splitext(gt_name)[0]

        #read index txt file
        txt_path = os.path.join(self.txt_path, gt_name + '.txt')
        # print('index_txt',txt_path)

        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read().rstrip('\n')  # 读取整个文件内容
            # print('psf',content)
        #根据self.psf_path和content路径连在一起读取pth文件
        psf_path = os.path.join(self.psf_path, content)
        psf = torch.load(psf_path, map_location='cpu')
        # psf = torch.randn(64, 51, 51, 3)  # 示例PSF张量，H=40, W=60
        fwhm = calculate_fwhm(psf)  # 形状为 [N, 2]
        # print('fwhm',fwhm.shape)


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, fov = paired_random_crop_fov(img_gt, img_lq, fov,gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq, fov = augment_fov([img_gt, img_lq,fov], self.opt['use_hflip'], self.opt['use_rot'])
        #fov numpy (h,w,1) ->  tensor [1,h,w]
        fov = torch.from_numpy(fov.copy().transpose(2, 0, 1)).float()
        #[1,h,w] -> [h,w]
        fov = fov.squeeze(0)
        energy_map_crop = create_energy_map_from_fov(fov, fwhm)
        #按通道归一化
        energy_map_crop = energy_map_crop / energy_map_crop.max()
        #[h,w,2] -> [2,h,w]
        energy_map_crop = energy_map_crop.permute(2, 0, 1)
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        energy_map_crop = self.pixel_unshuffle(energy_map_crop)

        # fov = torch.from_numpy(fov.copy().transpose(2, 0, 1)).float()
        #img_lq fov concat
        # img_lq = torch.cat([img_lq, energy_map_crop], 0)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        #print shape
        # print('img_lq',img_lq.shape,'img_gt',img_gt.shape)
        return {'lq': img_lq, 'gt': img_gt, 'psf':energy_map_crop, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)