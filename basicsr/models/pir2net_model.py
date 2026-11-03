
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.models.image_restoration_model import ImageRestorationModel

@MODEL_REGISTRY.register()
class PIR2Net_Model(ImageRestorationModel):
    """Base Deblur model for single image deblur."""

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.psf = data['psf'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds, psf_preds = self.net_g(self.lq, self.psf)
        if not isinstance(preds, list):
            preds = [preds]
            psf_preds = [psf_preds]

        self.output = preds[-1]
        self.psfout = psf_preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred, psf_pred in zip(preds, psf_preds):
                l_pix += self.cri_pix(pred,  psf_pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        # if self.cri_perceptual:
        #
        #
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred, xxx = self.net_g(self.lq[i:j, :, :, :], self.psf[i:j, :, :, :])
                if isinstance(pred, list):
                    pred = pred[-1]
                # print('pred .. size', pred.size())
                if isinstance(xxx, list):
                    xxx = xxx[-1]
                # print('pred .. size', pred.size())
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def single_image_inference(self, img, psf, save_path):
        self.feed_data(data={'lq': img.unsqueeze(dim=0),'psf':psf.unsqueeze(dim=0)})

        if self.opt['val'].get('grids', False):
            self.grids()

        self.test()

        if self.opt['val'].get('grids', False):
            self.grids_inverse()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_path)

    # def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
    #     logger = get_root_logger()
    #     # logger.info('Only support single GPU validation.')
    #     import os
    #     if os.environ['LOCAL_RANK'] == '0':
    #         return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
    #     else:
    #         return 0.

    # def nondist_validation(self, dataloader, current_iter, tb_logger,
    #                        save_img, rgb2bgr, use_image):
    #     dataset_name = dataloader.dataset.opt['name']
    #     with_metrics = self.opt['val'].get('metrics') is not None
    #     use_pbar = self.opt['val'].get('pbar', False)
    #     if with_metrics:
    #         self.metric_results = {
    #             metric: 0
    #             for metric in self.opt['val']['metrics'].keys()
    #         }
    #     pbar = tqdm(total=len(dataloader), unit='image')

    #     cnt = 0

    #     for idx, val_data in enumerate(dataloader):
    #         img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
    #         # if img_name[-1] != '9':
    #         #     continue
    #         # print('val_data .. ', val_data['lq'].size(), val_data['gt'].size())
    #         self.feed_data(val_data)
    #         if self.opt['val'].get('grids', False):
    #             self.grids()

    #         self.test()

    #         if self.opt['val'].get('grids', False):
    #             self.grids_inverse()

    #         visuals = self.get_current_visuals()
    #         sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
    #         if 'gt' in visuals:
    #             gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
    #             del self.gt

    #         # tentative for out of GPU memory
    #         del self.lq
    #         del self.psf
    #         del self.output
    #         torch.cuda.empty_cache()

    #         if save_img:

    #             if self.opt['is_train']:

    #                 save_img_path = osp.join(self.opt['path']['visualization'],
    #                                          img_name,
    #                                          f'{img_name}_{current_iter}.png')

    #                 save_gt_img_path = osp.join(self.opt['path']['visualization'],
    #                                          img_name,
    #                                          f'{img_name}_{current_iter}_gt.png')
    #             else:

    #                 save_img_path = osp.join(
    #                     self.opt['path']['visualization'], dataset_name,
    #                     f'{img_name}.png')
    #                 save_gt_img_path = osp.join(
    #                     self.opt['path']['visualization'], dataset_name,
    #                     f'{img_name}_gt.png')

    #             imwrite(sr_img, save_img_path)
    #             imwrite(gt_img, save_gt_img_path)

    #         if with_metrics:
    #             # calculate metrics
    #             opt_metric = deepcopy(self.opt['val']['metrics'])
    #             if use_image:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(
    #                         metric_module, metric_type)(sr_img, gt_img, **opt_)
    #             else:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(
    #                         metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

    #         pbar.update(1)
    #         pbar.set_description(f'Test {img_name}')
    #         cnt += 1
    #         # if cnt == 300:
    #         #     break
    #     pbar.close()

    #     current_metric = 0.
    #     if with_metrics:
    #         for metric in self.metric_results.keys():
    #             self.metric_results[metric] /= cnt
    #             current_metric = self.metric_results[metric]

    #         self._log_validation_metric_values(current_iter, dataset_name,
    #                                            tb_logger)
    #     return current_metric


    # def _log_validation_metric_values(self, current_iter, dataset_name,
    #                                   tb_logger):
    #     log_str = f'Validation {dataset_name},\t'
    #     for metric, value in self.metric_results.items():
    #         log_str += f'\t # {metric}: {value:.4f}'
    #     logger = get_root_logger()
    #     logger.info(log_str)
    #     if tb_logger:
    #         for metric, value in self.metric_results.items():
    #             tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    # def get_current_visuals(self):
    #     out_dict = OrderedDict()
    #     out_dict['lq'] = self.lq.detach().cpu()
    #     out_dict['result'] = self.output.detach().cpu()
    #     if hasattr(self, 'gt'):
    #         out_dict['gt'] = self.gt.detach().cpu()
    #     return out_dict

    # def save(self, epoch, current_iter):
    #     self.save_network(self.net_g, 'net_g', current_iter)
    #     self.save_training_state(epoch, current_iter)

