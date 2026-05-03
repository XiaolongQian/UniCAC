# UniCAC

**Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis**

> Xiaolong Qian\* · Qi Jiang\* · Yao Gao · Lei Sun† · Zhonghua Yi · Kailun Yang · Luc Van Gool · Kaiwei Wang†  
> \* Equal contribution &nbsp;|&nbsp; † Corresponding authors  
> Zhejiang University · INSAIT, Sofia University "St. Kliment Ohridski" · Hunan University  


## ✨ Summary

- **Comprehensive CAC benchmark**: large-scale dataset and benchmark for photographic cameras, built via automatic optical design to cover diverse optical aberrations.
- **Optical Degradation Evaluator (ODE)**: framework for objectively quantifying optical aberrations and CAC task difficulty.
- **Key design insights**: systematic comparison of 24 methods reveals three dominant factors for CAC performance — **prior utilization**, **network architecture**, and **training strategy** — and analyzes their respective roles.


## ✅ TODO

UniCAC is a universal benchmark for Computational Aberration Correction (CAC) emphasizing optical-physics intuition. Assets will be released in the following stages:

### 1. Data & Optical Assets
- [ ] Release Train / Val / Test datasets
- [ ] Release Zemax files (`.zmx`)
- [ ] Release simulated PSF assets

### 2. Evaluation Metrics
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
