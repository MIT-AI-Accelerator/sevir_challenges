# Radar Nowcast Challenge

The radar nowcast challenge is to generate future radar imagery given previous radar and satellite imagery as input.


This challenge is still in development.  See [RadarNowcastChallenge notebook](RadarNowcastBenchmarks.ipynb) for a description of the datasets, problem, baseline model, and metrics.


## Leaderboard
| Model | MSE | CSI | LPIPS  | Reference |
|--------------------|----------------------------------|----------------------------|----------------------|
| U-Net (MSE Loss) | 466.64 | | 0.3934 | [Veillette et al. 2020](https://proceedings.neurips.cc//paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf)
| U-Net (Content Loss) | 497.26 | | 0.6195 | [Veillette et al. 2020](https://proceedings.neurips.cc//paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf)
| U-Net (cGAN + MAE) | 738.41 | | 0.3498 | [Veillette et al. 2020](https://proceedings.neurips.cc//paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf)
| Optical flow (baseline) | 791.70 | | 0.2836 | [Veillette et al. 2020](https://proceedings.neurips.cc//paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf)
| Persistence (baseline) | 1339.17 | | 0.3176 | [Veillette et al. 2020](https://proceedings.neurips.cc//paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf)
