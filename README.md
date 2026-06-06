# Federated Learning Framework

Python implementation of **FedAF (CVPR 2024)** (Aggregation-Free Federated Learning) and **FedAvg** (Federated Averaging) Frameworks.

## Overview

Federated Learning enables collaborative model training across decentralized devices holding local data, without exchanging the data itself. This framework allows to compare FedAF and FedAvg in terms of performance and efficiency.

## Features

- **FedAF Implementation**: Benchmarking Aggregation-Free Federated Learning using synthetic data condensation.
- **FedAvg Implementation**: Standard Federated Averaging algorithm for comparison.
- **Dataset Support**: MNIST, FMNIST (Fashion-MNIST), CIFAR-10 and CIFAR-100 with non-IID (Dirichlet label-skew) partitioning. Adding a dataset is a one-line entry in `DATASET_REGISTRY` (`config.py`).
- **Single-source configuration**: Every setting lives in `config.yaml` and propagates to FedAF, FedAvg and the plotter — see [Configuration](#configuration).
- **Restartability**: Runs resume where they stopped — completed rounds and, within a round, completed phases are skipped using saved artifacts — see [Restartability](#restartability).

## Requirements

- Python 3.x
- torch
- torchvision
- numpy
- matplotlib
- pyyaml

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/tejarshaarigila/FedAF.git
cd Federated-Learning-Framework
pip install torch torchvision numpy matplotlib pyyaml
```

## Configuration

All settings live in a single file, **`config.yaml`**, and are loaded by `config.py`
into one `Config` object shared by FedAF, FedAvg and the plotter. Change a value
once (e.g. the dataset, client count or a hyperparameter) and it propagates
everywhere — there is no per-script duplication.

```yaml
dataset: CIFAR10        # MNIST | FMNIST | CIFAR10 | CIFAR100
model: ConvNet
num_clients: 10
alpha: 0.1              # Dirichlet label-skew
rounds: 20
resume: true
fedaf:  { ipc: 50, iteration: 1000, global_epochs: 500, loc_cdc: 0.0001, loc_lgkm: 0.01, ... }
fedavg: { local_epochs: 10, lr: 0.01, batch_size: 64 }
paths:  { data: ./data, logits: ./logits, models: ./models, result: ./result, plots: ./plots }
```

Per-dataset constants (channels, classes, image size, normalisation, default
`lr_img`) come from `DATASET_REGISTRY` in `config.py`. The synthetic-image
learning rate defaults to the paper's per-dataset value (CIFAR-10 = 1.0,
CIFAR-100 = 0.1, MNIST/FMNIST = 0.2) unless `fedaf.lr_img` is set.

A few common fields can be overridden per run on the command line (the YAML
remains the source of truth), and a different config file can be selected:

```bash
python main_fedaf.py --dataset FMNIST --rounds 10 --ipc 20
python main_fedaf.py --config experiments/cifar100.yaml
FEDAF_CONFIG=experiments/cifar100.yaml python main_fedaf.py
```

## Usage

### Training

```bash
python main_fedaf.py     # FedAF
python main_fedavg.py    # FedAvg
```

### Evaluation

```bash
python main_plot.py      # plots FedAF vs FedAvg test accuracy per round -> paths.plots
```

## Restartability

When `resume: true` (the default), an interrupted run continues instead of
starting over. Progress is detected from artifacts on disk:

- A round whose global model already exists is skipped entirely.
- Within a round, each phase is skipped if its output is present: per-client
  logits `v_{k,c}` (Phase 1), the aggregated `V`, per-client condensed data and
  soft labels `r_{k,c}` (Phase 2), the aggregated `R`, and the trained server
  model.
- FedAvg resumes from its last saved round checkpoint.

Set `resume: false` (or pass `--no-resume`) to force a clean run. To redo a
specific round, delete its `*_global_model_<round>.pth` checkpoint.

## Directory Structure

- `config.yaml`: Single source of truth for all settings.
- `config.py`: Loads the config + dataset registry; restart helpers.
- `client/`: Client-side training logic.
- `server/`: Server-side aggregation and coordination.
- `utils/`: Utility functions for data handling and processing.
- `main_fedaf.py`: FedAF training driver.
- `main_fedavg.py`: FedAvg training driver.
- `main_plot.py`: Plots and compares results.

## Citation

If you use this framework in your research, please cite the following paper:

```
@inproceedings{Wang2024FedAF,
  title     = {An Aggregation-Free Federated Learning for Tackling Data Heterogeneity},
  author    = {Wang, Shixiong and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
  url       = {https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_An_Aggregation-Free_Federated_Learning_for_Tackling_Data_Heterogeneity_CVPR_2024_paper.pdf}
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
