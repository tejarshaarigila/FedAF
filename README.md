# Federated Learning Framework

This repository provides a Python implementation of **FedAF** (Aggregation-Free Federated Learning) and **FedAvg** (Federated Averaging) algorithms. It facilitates benchmarking these federated learning methods using the MNIST and CIFAR-10 datasets under non-IID conditions.

## Overview

Federated Learning enables collaborative model training across decentralized devices holding local data, without exchanging the data itself. This framework allows to compare FedAF and FedAvg in terms of performance and efficiency.

## Features

- **FedAF Implementation**: Benchmarking Aggregation-Free Federated Learning using synthetic data condensation.
- **FedAvg Implementation**: Standard Federated Averaging algorithm for comparison.
- **Dataset Support**: Utilizes MNIST and CIFAR-10 datasets with non-IID data partitioning.

## Requirements

- Python 3.x
- torch
- torchvision
- numpy
- matplotlib
- scikit-learn

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/tejarshaarigila/Federated-Learning-Framework.git
cd Federated-Learning-Framework
```

## Usage

### Training

To train models using FedAF or FedAvg, run the respective scripts:

```bash
# For FedAF
python main_fedaf.py

# For FedAvg
python main_fedavg.py
```

### Evaluation

After training, evaluate the models using:

```bash
python main_plot.py
```

This script generates plots comparing the performance of FedAF and FedAvg.

## Directory Structure

- `client/`: Contains client-side training logic.
- `server/`: Contains server-side aggregation and coordination.
- `utils/`: Utility functions for data handling and processing.
- `main_fedaf.py`: Script to initiate FedAF training.
- `main_fedavg.py`: Script to initiate FedAvg training.
- `main_plot.py`: Script to plot and compare results.

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
