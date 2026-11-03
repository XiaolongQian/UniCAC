# UniCAC

**Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis**

Xiaolong Qian* · Qi Jiang* · Yao Gao · Lei Sun† · Zhonghua Yi · Kailun Yang · Luc Van Gool · Kaiwei Wang†  
*Equal contribution; †Corresponding authors.*  
Zhejiang University · INSAIT, Sofia University · Hunan University

A BasicSR-based framework for computational aberration correction. It provides **9 training configurations** and **25-group test configurations for 10 methods**.

- [Dataset](https://huggingface.co/datasets/Residual/UniCAC)
- [Pretrained weights — upload pending](https://huggingface.co/Residual/UniCAC)
- [Installation](#installation) · [Data-and-weight preparation](#data-and-weight-preparation) · [Training](#training) · [Inference](#inference)

The dataset has been released. Ten restoration checkpoints and two FeMaSR training priors are prepared for a separate Hugging Face upload; they are not included in this GitHub repository.

## Installation

The reference environment is **Linux x86_64, Python 3.9, PyTorch 1.13.1, torchvision 0.14.1, CUDA 11.6, and mmcv-full 1.7.2**. Use an NVIDIA GPU and a compatible NVIDIA driver. The CUDA runtime is supplied by the PyTorch wheels; the default installation uses prebuilt MMCV operators.

```bash
git clone https://github.com/XiaolongQian/UniCAC.git
cd UniCAC

conda create -n unicac python=3.9 -y
conda activate unicac
python -m pip install pip==24.2 setuptools==75.1.0 wheel==0.44.0
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

`requirements.txt` contains the required packages pinned to the reference versions, including the PyTorch download index and matching MMCV wheel source. It excludes unrelated packages from the original environment, such as mmedit, torchaudio, pyiqa, and profiling tools. The final editable-install command makes the local `basicsr` package available to the native scripts.

All commands below run from the repository root. `CUDA_VISIBLE_DEVICES=0` selects one GPU; set `num_gpu: 1` in the YAML when using a single visible GPU. Do not install a separate upstream BasicSR package into this environment.

## Data and weight preparation

### Dataset

Download the data from [Residual/UniCAC](https://huggingface.co/datasets/Residual/UniCAC). The released directories may have `_release` suffixes. Either edit the YAML paths to their actual locations or arrange them as follows:

```text
UniCAC/
├── datasets/
│   ├── CACBench_Data/
│   │   ├── train/{ab,gt,label}/
│   │   ├── val/{ab,gt,label}/
│   │   └── test_sample/
│   │       ├── L1/<lens-name>/{ab,gt}/
│   │       ├── L2/<lens-name>/{ab,gt}/
│   │       ├── L3/<lens-name>/{ab,gt}/
│   │       ├── L4/<lens-name>/{ab,gt}/
│   │       └── L5/<lens-name>/{ab,gt}/
│   ├── CACBench_Lib/
│   │   ├── Train_Lib/{psf,psf_npy}/
│   │   ├── Val_Lib/{psf,psf_npy}/
│   │   └── Test_Lib/Level_1 ... Level_5/psf_npy/
│   └── fov_map_1280_1920.npy
├── options/train/<method>/train.yml
└── options/test/<method>/test.yml
```

Image pairs follow each dataset's filename rules. PART and PI2RNet training require image-to-PSF label files. Their test datasets identify PSFs from the lens directory name. PI2RNet also needs the FOV map. FOV/PSF preprocessing retains the original 1280×1920 coordinate convention; other resolutions require compatible preprocessing.

### Pretrained models

Weights will be available at **[Residual/UniCAC](https://huggingface.co/Residual/UniCAC)** after the separate upload. Download the method folder, then set `path.pretrain_network_g` in its test YAML. For example:

```yaml
path:
  pretrain_network_g: /path/to/dfunet/DFUNet.pth
  strict_load_g: true
  param_key_g: params
```

Paths are read directly from YAML and are relative to the working directory unless absolute. The current test configurations use `../huggingface/<method>/<Model>.pth`; keep that layout or edit the paths.

| Method | Config directory | Restoration checkpoint | Train config |
|---|---|---|---|
| DFUNet | `dfunet` | `DFUNet.pth` | Yes |
| FeMaSR | `femasr` | `FeMaSR.pth` | Yes |
| FOV-KPN | `fovkpn` | `FOVKPN.pth` | Yes |
| PART | `PART` | `PART.pth` | Yes |
| SwinIR | `swinir` | `SwinIR.pth` | Yes |
| PI2RNet | `PIR2Net` | `PI2RNet.pth` | Yes |
| MIMOUnetPlus | `mimo_unet` | `MIMOUnetPlus.pth` | Yes |
| NAFNet | `NAFNet` | `NAFNet.pth` | Yes |
| Restormer | `restormer` | `Restormer.pth` | Not included |
| Uformer | `uformer` | `Uformer.pth` | Yes |

FeMaSR training additionally requires **HQP_v2.pth** and **HQP_d_v2.pth**. Set `pretrain_network_hq` and `pretrain_network_d` in `options/train/femasr/train.yml`.

## Training

Edit the method's YAML under `options/train/`: dataset paths, `num_gpu`, batch size, total iterations, losses, and checkpoint paths. Your training settings are taken directly from that YAML. Horizontal-flip augmentation accepts both `use_hflip` and the legacy `use_flip` key.

### Start a new training run

Leave `path.pretrain_network_g` and `path.resume_state` unset for a new run. For FeMaSR, keep the required prior paths configured in its training YAML.

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/dfunet/train.yml
```

Use the corresponding config for another available method:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/PART/train.yml
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/NAFNet/train.yml
```

### Fine-tune a pretrained model

Set the following fields in the same training YAML, then run the normal training command:

```yaml
path:
  pretrain_network_g: /path/to/DFUNet.pth
  strict_load_g: true
  param_key_g: params
  resume_state: ~
```

This loads model parameters and starts a new optimizer/scheduler history.

## Inference

Edit the dataset and checkpoint paths in the method's YAML under `options/test/`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/dfunet/test.yml
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/PART/test.yml
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/restormer/test.yml
```

Each test configuration contains **25 groups: five difficulty levels with five lens groups per level**. The native test script processes every configured group. Results and logs are written under `results/<name>/`; restored images are grouped below `results/<name>/visualization/` by dataset name. The current paired datasets require GT images, and PSF-based methods require their corresponding PSF assets.

Evaluation is controlled by `val.metrics` in the test YAML. Set `metrics: ~` for image output only; retain/configure PSNR, SSIM, and LPIPS when evaluation is required. To run a subset, keep only the desired dataset entries in a copy of the YAML.

### Original inference defaults

| Methods | Default behavior |
|---|---|
| DFUNet, FOV-KPN, PART, SwinIR, PI2RNet, MIMOUnetPlus, NAFNet | Full-image forward pass |
| Restormer | Original overlapping grids, `crop_size: 128`, `max_minibatch: 8` |
| Uformer | Original overlapping grids, `crop_size: 256`, `max_minibatch: 8` |
| FeMaSR | Original padding to a multiple of 64 and 8000×8000 area threshold for full-image/tiled inference, cropped back to input size |

Inference uses the per-method settings listed above.

## Citation

```bibtex
@article{qian2026towards,
  title={Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis},
  author={Qian, Xiaolong and Jiang, Qi and Gao, Yao and Sun, Lei and Yi, Zhonghua and Yang, Kailun and Van Gool, Luc and Wang, Kaiwei},
  journal={arXiv preprint arXiv:2603.12083},
  year={2026}
}
```

## Acknowledgements and license

This implementation is based on BasicSR and incorporates the selected methods' upstream implementations. We thank their authors. Retain the source copyright notices and consult [LICENSE.txt](LICENSE.txt), [third-party notices](LICENSE/README.md), and the corresponding upstream licenses for their respective terms. BasicSR's original citation is preserved in [LICENSE/CITATION_BasicSR.cff](LICENSE/CITATION_BasicSR.cff).
