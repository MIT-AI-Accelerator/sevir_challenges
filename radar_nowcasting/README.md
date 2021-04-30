# Radar Nowcast Challenge

The radar nowcast challenge is to generate future radar imagery given previous radar and satellite imagery as input.


This challenge is still in development.  See [RadarNowcastChallenge notebook](RadarNowcastBenchmarks.ipynb) for a description of the datasets, problem, baseline model, and metrics.


## Leaderboard
| Model | MAE | mCSI | Reference |
|-------|-----|------|-----------|
| U-Net (MSE Loss) | 11.36 | 0.2530 | [Veillette et al. 2020](https://proceedings.neurips.cc//paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf)|
| U-Net (MSE + Content Loss) |  | [Veillette et al. 2020](https://proceedings.neurips.cc//paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf)|
| U-Net (cGAN + MAE) | 11.20 | 0.4690| [Veillette et al. 2020](https://proceedings.neurips.cc//paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf)|
| Persistence (baseline) |  |  | [Veillette et al. 2020](https://proceedings.neurips.cc//paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf)|
