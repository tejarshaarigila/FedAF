# **Framework to Benchmark [Federated Aggregation Free](https://doi.org/10.48550/arXiv.2404.18962) and [Federated Averaging] (https://doi.org/10.48550/arXiv.1602.05629)**

## **FedAVG Implementation**

### **1. Parameters Overview**

#### **Dataset Parameters**
- **`dataset`**: The dataset used for federated learning. Options:
  - `'MNIST'`
  - `'CIFAR10'`
- **`channel`**: Number of image channels (auto-set based on dataset):
  - `1`: For grayscale (e.g., MNIST).
  - `3`: For RGB (e.g., CIFAR10).
- **`num_classes`**: Total classes in the dataset (e.g., `10` for CIFAR10).
- **`im_size`**: Image size in pixels (e.g., `(28, 28)` for MNIST, `(32, 32)` for CIFAR10).

#### **Model Parameters**
- **`model`**: Model architecture used for local training:
  - `'ConvNet'`: Convolutional Neural Network.
  - `'ResNet'`: Residual Network.
- **`device`**: Training device:
  - `'cuda'`: If a GPU is available.
  - `'cpu'`: Otherwise.

#### **Federated Learning Parameters**
- **`num_clients`**: Number of participating clients.
- **`alpha`**: Dirichlet distribution parameter controlling client data heterogeneity.

#### **Training Parameters**
- **`local_epochs`**: Number of local training epochs per client.
- **`lr`**: Learning rate for the optimizer.
- **`batch_size`**: Batch size for local training.
- **`num_rounds`**: Total number of server-client communication rounds.

---

### **2. Default Configuration**

| **Parameter**      | **Default Value**    |
|---------------------|----------------------|
| `dataset`          | `'MNIST'`           |
| `model`            | `'ConvNet'`         |
| `device`           | `'cuda'` or `'cpu'` |
| `num_clients`      | `5`                 |
| `alpha`            | `0.1`               |
| `local_epochs`     | `10`                |
| `lr`               | `0.01`              |
| `batch_size`       | `64`                |
| `num_rounds`       | `20`                |

---

### **3. Customizing Parameters**
To customize these parameters:
1. Update the `ARGS` class in the `main_fedavg.py` script.

---

## **FedAF Implementation**

### **1. Parameters Overview**

#### **Dataset Parameters**
- **`dataset`**: Dataset for FedAF simulation. Options:
  - `'MNIST'`
  - `'CIFAR10'`
- **`data_path`**: Directory path for dataset storage.
- **`channel`**: Number of image channels:
  - `1`: For MNIST.
  - `3`: For CIFAR10.
- **`num_classes`**: Number of dataset classes.
- **`im_size`**: Image dimensions.
- **`mean` and `std`**: Normalization values specific to the dataset.

#### **Federated Learning Parameters**
- **`num_partitions`**: Number of clients.
- **`alpha`**: Dirichlet distribution parameter.

#### **Training and Evaluation Parameters**
- **`Iteration`**: Local training steps per client.
- **`ipc`**: Instances per class for synthetic data condensation.
- **`lr_img`**: Learning rate for synthetic image optimization.
- **`steps`**: Frequency of global aggregation.
- **`temperature`**: Softmax temperature for logit aggregation.
- **`gamma`**: Momentum coefficient for logit aggregation.

#### **Directories and Paths**
- **`logits_dir`**: Path for aggregated logits storage.
- **`save_image_dir`**: Path to save synthetic images.
- **`save_path`**: Final results directory.

---

### **2. Default Configuration**

| **Parameter**      | **Default Value**    |
|---------------------|----------------------|
| `dataset`          | `'MNIST'`           |
| `model`            | `'ConvNet'`         |
| `device`           | `'cuda'` or `'cpu'` |
| `num_partitions`   | `15`                |
| `alpha`            | `0.1`               |
| `Iteration`        | `1000`              |
| `ipc`              | `50`                |
| `lr_img`           | `1`                 |
| `steps`            | `500`               |
| `temperature`      | `2.0`               |
| `gamma`            | `0.9`               |
| `eval_mode`        | `'SS'`              |

---

### **3. Customizing Parameters**
To modify these configurations:
1. Update the `ARGS` class in the `main_fedaf.py` script.

---

## **How to Run `main_plot.py`**

### **1. Purpose**
- **Evaluate Model Accuracy**:
  - Load saved global models from federated learning experiments.
  - Compute and compare average test accuracy.
- **Generate Plots**:
  - Test accuracy vs. communication rounds for methods (e.g., `FedAF` vs. `FedAvg`).

---

### **2. Requirements**
- **Python Libraries**:
  - `torch`, `matplotlib`, `numpy`, `argparse`, `multiprocessing`, `torchvision`.
- **Datasets**:
  - Supported: `MNIST`, `CIFAR10`.
- **Model Checkpoints**:
  - Naming: `{method}_global_model_{round_number}.pth`.

---

### **3. Usage**

#### **Basic Command**
```bash
python main_plot.py --dataset CIFAR10 --model ConvNet --methods fedaf fedavg
```

#### **Command-Line Arguments**

| **Argument**         | **Type**  | **Default**    | **Description**                                                      |
|-----------------------|-----------|----------------|----------------------------------------------------------------------|
| `--dataset`          | `str`     | `CIFAR10`      | Dataset (`MNIST` or `CIFAR10`).                                      |
| `--model`            | `str`     | `ConvNet`      | Model architecture (e.g., `ConvNet`).                                |
| `--device`           | `str`     | Auto-detect    | Computation device (`cuda` or `cpu`).                                |
| `--test_repeats`     | `int`     | `5`            | Test repetition count for averaging.                                 |
| `--num_users`        | `int`     | `10`           | Number of clients/users.                                             |
| `--alpha_dirichlet`  | `float`   | `0.1`          | Dirichlet parameter for data heterogeneity.                          |
| `--methods`          | `list`    | `['fedaf', 'fedavg']` | Methods to compare.                                              |
| `--model_base_dir`   | `str`     | `/home/models` | Base directory for trained models.                                   |
| `--save_dir`         | `str`     | `/home/plots/` | Directory to save plots.                                             |

---

### **4. Example Command**
```bash
python main_plot.py --dataset MNIST --model ConvNet --methods fedaf fedavg --num_users 10 --alpha_dirichlet 0.1
```

#### **Output**
- **Plot Name**: `MNIST_ConvNet_C10_alpha0.1.png`
- **Saved At**: `/home/plots/`

---
