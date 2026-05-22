# UniCAC

**Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis**

> Xiaolong Qian\* · Qi Jiang\* · Yao Gao · Lei Sun† · Zhonghua Yi · Kailun Yang · Luc Van Gool · Kaiwei Wang†  
> \* Equal contribution &nbsp;|&nbsp; † Corresponding authors  
> Zhejiang University · INSAIT, Sofia University "St. Kliment Ohridski" · Hunan University  


## ✨ Summary

- **Comprehensive CAC benchmark**: large-scale dataset and benchmark for photographic cameras, built via automatic optical design to cover diverse optical aberrations.
- **Optical Degradation Evaluator (ODE)**: framework for objectively quantifying optical aberrations and CAC task difficulty.
- **Key design insights**: systematic comparison of 24 methods reveals three dominant factors for CAC performance — **prior utilization**, **network architecture**, and **training strategy** — and analyzes their respective roles.


## 📰 News

- **2026-05-22**: The CACBench benchmark dataset has been released on [Hugging Face](https://huggingface.co/datasets/Residual/UniCAC).

The released dataset contains four parts:

```text
CACBench_Data_release/
CACBench_Lib_release/
CACBench_Zmx_release/
CACBench_Checker_release/
```

- `CACBench_Data_release/`: image data and labels for training, validation, and testing.
- `CACBench_Lib_release/`: degradation/lens library files used by the dataset.
- `CACBench_Zmx_release/`: optical design files, including ZMX/ZDA files.
- `CACBench_Checker_release/`: checker files for evaluation and visualization.


## ✅ TODO

### 1. Data & Optical Assets
- [x] Release Train / Val / Test datasets
- [x] Release degradation/lens library files
- [x] Release Zemax files (`.zmx`, `.zda`)
- [x] Release simulated PSF assets

### 2. Evaluation Metrics
- [x] Release checker files for evaluation and visualization
- [ ] Release ODE calculation script
- [ ] Release Overall Performance (O.P.) evaluation script

### 3. Unified Framework
- [ ] Release unified codebase integrating most of the evaluated baseline models




## 🎓 Citation

If you find this work useful, please consider citing:

```bibtex
@article{qian2026towards,
  title={Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis},
  author={Qian, Xiaolong and Jiang, Qi and Gao, Yao and Sun, Lei and Yi, Zhonghua and Yang, Kailun and Van Gool, Luc and Wang, Kaiwei},
  journal={arXiv preprint arXiv:2603.12083},
  year={2026}
}
```
