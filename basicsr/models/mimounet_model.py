import torch
from torch.nn import functional as F
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img

from basicsr.utils.registry import MODEL_REGISTRY
from .image_restoration_2loss_model import ImageRestoration_2loss_Model


@MODEL_REGISTRY.register()
class MIMOUnet_Model(ImageRestoration_2loss_Model):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('pixel_opt2'):
            self.cri_pix2 = build_loss(train_opt['pixel_opt2']).to(self.device)
        else:
            self.cri_pix2 = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # self.output = self.net_g(self.lq)
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        gt_1_4 = F.interpolate(self.gt, scale_factor=0.25, mode='bicubic')
        gt_1_2 = F.interpolate(self.gt, scale_factor=0.5, mode='bicubic')
        gts = [gt_1_4, gt_1_2, self.gt]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # # pixel loss
        # if self.cri_pix:
        #     l_pix = self.cri_pix(self.output, self.gt)
        #     l_total += l_pix
        #     loss_dict['l_pix'] = l_pix
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            # for pred in preds:
            #     l_pix += self.cri_pix(pred, self.gt)
            for pred, gt in zip(preds, gts):
                l_pix += self.cri_pix(pred, gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # pixel loss2
        if self.cri_pix2:
            l_pix2 = 0.
            # for pred in preds:
            #     l_pix2 += self.cri_pix2(pred, self.gt)
            for pred, gt in zip(preds, gts):
                # if torch.isnan(pred).any() or torch.isinf(pred).any():
                #     raise ValueError("输入张量包含非法值（NaN 或 Inf）")
                # if torch.isnan(gt).any() or torch.isinf(gt).any():
                #     raise ValueError("输入张量包含非法值（NaN 或 Inf）")
                # #print 是否为float
                # print('pred type ... ', pred.dtype,'gt shape ... ', gt.dtype)
                # print('pred shape ... ', pred.shape,'gt shape ... ', gt.shape)
                l_pix2 += self.cri_pix2(pred.cpu(), gt.cpu())
                l_pix2 = l_pix2.to(self.device)
            l_total += l_pix2
            loss_dict['l_pix2'] = l_pix2
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)