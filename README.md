# **FedAVG Implementation**

### **Parameters Overview**

#### **Dataset Parameters**
- `dataset`: The dataset used for federated learning. Options:
  - `'MNIST'`
  - `'CIFAR10'`
- `channel`: Automatically set based on the chosen dataset. Indicates the number of image channels.
  - `1` for grayscale datasets like MNIST.
  - `3` for RGB datasets like CIFAR10.
- `num_classes`: Number of classes in the dataset (e.g., `10` for CIFAR10).
- `im_size`: Image size in pixels (e.g., `(28, 28)` for MNIST, `(32, 32)` for CIFAR10).

#### **Model Parameters**
- `model`: The model architecture used for local training. Options:
  - `'ConvNet'` for a Convolutional Neural Network.
  - `'ResNet'` for a Residual Network.
- `device`: Determines the computation device.
  - `'cuda'` if a GPU is available.
  - `'cpu'` otherwise.

#### **Federated Learning Parameters**
- `num_clients`: Number of clients participating in the federated learning setup.
- `alpha`: Parameter for the Dirichlet distribution to control data heterogeneity among clients.

#### **Training Parameters**
- `local_epochs`: Number of local training epochs performed by each client.
- `lr`: Learning rate for the optimizer during local training.
- `batch_size`: Size of data batches used during training.
- `num_rounds`: Total number of communication rounds between the server and clients.

---

### **Default Configuration**
Below is the default configuration used in the provided FedAvg implementation:

| Parameter         | Value               |
|--------------------|---------------------|
| `dataset`         | `'MNIST'`          |
| `model`           | `'ConvNet'`         |
| `device`          | `'cuda'` or `'cpu'` |
| `num_clients`     | `5`                 |
| `alpha`           | `0.1`               |
| `local_epochs`    | `10`                |
| `lr`              | `0.01`              |
| `batch_size`      | `64`                |
| `num_rounds`      | `20`                |

---

### **Customizing Parameters**
To customize these parameters:
1. Modify the values in the `ARGS` class in main_fedavg.py script.

---

# **FedAF Implementation**

### **Parameters Overview**

#### **Dataset Parameters**
- `dataset`: The dataset used in the FedAF simulation. Options:
  - `'MNIST'`
  - `'CIFAR10'`
- `data_path`: Directory path where the dataset is stored.
- `channel`: Number of image channels (automatically set based on the dataset).
  - `1` for MNIST.
  - `3` for CIFAR10.
- `num_classes`: Number of classes in the dataset.
- `im_size`: Dimensions of the images.
- `mean` and `std`: Dataset-specific normalization values.

#### **Federated Learning Parameters**
- `num_partitions`: Number of clients participating in federated learning.
- `alpha`: Dirichlet distribution parameter to control data heterogeneity across clients.

#### **Training and Evaluation Parameters**
- `Iteration`: Number of local training steps per client.
- `ipc`: Instances per class used in synthetic data condensation.
- `lr_img`: Learning rate for image optimization during condensation.
- `steps`: Frequency of global aggregation steps.
- `temperature`: Softmax temperature used during logit aggregation.
- `gamma`: Momentum coefficient for logit aggregation.

#### **Directories and Paths**
- `logits_dir`: Directory to store aggregated logits (`V` and `R` values).
- `save_image_dir`: Directory to save synthetic images generated during condensation.
- `save_path`: Directory to save final results.

---

### **Default Configuration**
The default configuration for FedAF is as follows:

| Parameter         | Value               |
|--------------------|---------------------|
| `dataset`         | `'MNIST'`           |
| `model`           | `'ConvNet'`         |
| `device`          | `'cuda'` or `'cpu'` |
| `num_partitions`  | `15`                |
| `alpha`           | `0.1`               |
| `Iteration`       | `1000`              |
| `ipc`             | `50`                |
| `lr_img`          | `1`                 |
| `steps`           | `500`               |
| `temperature`     | `2.0`               |
| `gamma`           | `0.9`               |
| `eval_mode`       | `'SS'`              |

---

### **Customizing Parameters**
To modify the configuration:
1. Edit the values in the `ARGS` class in main_fedaf.py script.

---

# How to Run `main_plot.py`

The `main_plot.py` script is designed to evaluate saved models from federated learning experiments (FedAF, FedAvg, etc.) and generate performance plots across communication rounds. This guide provides step-by-step instructions for running the script.

---

### **Purpose of the Script**
- **Evaluate Model Accuracy**:
  - Load saved global models from different rounds.
  - Compute average test accuracy for specified methods.
- **Generate Plots**:
  - Plot test accuracy against communication rounds for comparison of methods (e.g., `FedAF` vs. `FedAvg`).

---

### **Requirements**
1. **Python Libraries**:
   - `torch`, `matplotlib`, `numpy`, `argparse`, `multiprocessing`
   - `torchvision` for loading datasets
2. **Dataset**:
   - Supported datasets: `CIFAR10`, `MNIST`.
3. **Models**:
   - Trained model checkpoints stored in the specified directory.
   - Naming convention: `{method}_global_model_{round_number}.pth`.

---

### **Usage**

#### **Basic Command**
Run the script from the terminal as follows:
```bash
python main_plot.py --dataset CIFAR10 --model ConvNet --methods fedaf fedavg
```

#### **Command-Line Arguments**
| Argument              | Type   | Default     | Description                                                                 |
|------------------------|--------|-------------|-----------------------------------------------------------------------------|
| `--dataset`           | `str`  | `CIFAR10`   | Dataset name (`CIFAR10` or `MNIST`).                                       |
| `--model`             | `str`  | `ConvNet`   | Model architecture (e.g., `ConvNet`).                                      |
| `--device`            | `str`  | Auto-detect | Device for computation (`cuda` or `cpu`).                                  |
| `--test_repeats`      | `int`  | `5`         | Number of times to repeat testing for averaging.                           |
| `--num_users`         | `int`  | `10`        | Number of clients/users.                                                   |
| `--alpha_dirichlet`   | `float`| `0.1`       | Dirichlet distribution parameter controlling data heterogeneity.           |
| `--methods`           | `list` | `['fedaf', 'fedavg']` | Methods to compare (e.g., `fedaf`, `fedavg`).                          |
| `--model_base_dir`    | `str`  | `/home/models` | Base directory where trained models are stored.                        |
| `--save_dir`          | `str`  | `/home/plots/` | Directory to save the resulting plots.                                 |

---

### **Example Use Case**
Command:
```bash
python main_plot.py --dataset MNIST --model ConvNet --methods fedaf fedavg --num_users 10 --alpha_dirichlet 0.1
```

Output:
- A plot named `MNIST_ConvNet_C10_alpha0.1.png` saved in `/home/plots/`.

---
