#!/bin/bash

#SBATCH -p intel
#SBATCH -N 1             
#SBATCH -n 1              
#SBATCH -c 10             
#SBATCH --mem=32G     
#SBATCH -t 48:00:00       
#SBATCH -J fedavg_exp
#SBATCH -o slurm-fedavg-%j.out

export OMP_NUM_THREADS=1  
export MKL_NUM_THREADS=1 

PYTHON_FILE_FEDAVG="main_fedavg.py"
DATASET="CIFAR10"
MODEL="ConvNet"
NUM_USERS=5
ALPHA_DIRICHLET=0.1
HONESTY_RATIO=1.0
NUM_ROUNDS=20

DATA_PATH="/home/t914a431/data"
MODEL_DIR="/home/t914a431/models/${DATASET}/${MODEL}/${NUM_USERS}/${HONESTY_RATIO}/"
LOGITS_DIR="/home/t914a431/logits"
SAVE_IMAGE_DIR="/home/t914a431/images"
SAVE_PATH="/home/t914a431/result"

mkdir -p "$DATA_PATH"
mkdir -p "$MODEL_DIR"
mkdir -p "$LOGITS_DIR"
mkdir -p "$SAVE_IMAGE_DIR"
mkdir -p "$SAVE_PATH"

echo "========================================"
echo "Starting FedAvg Experiment"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Number of Clients: $NUM_USERS"
echo "Alpha (Dirichlet): $ALPHA_DIRICHLET"
echo "Honesty Ratio: $HONESTY_RATIO"
echo "========================================"

echo "Pre-downloading the dataset..."
srun -n 1 -c 1 python3 -c "from torchvision import datasets; datasets.${DATASET}(root='${DATA_PATH}', train=True, download=True); datasets.${DATASET}(root='${DATA_PATH}', train=False, download=True)"

if [ $? -ne 0 ]; then
    echo "Error: Dataset download failed."
    exit 1
fi
echo "Dataset downloaded successfully."

echo "----------------------------------------"
echo "Running FedAvg"
echo "----------------------------------------"

srun -n 1 -c 10 python3 $PYTHON_FILE_FEDAVG

if [ $? -ne 0 ]; then
    echo "Error: FedAvg script failed."
    exit 1
fi
echo "FedAvg script completed successfully."

echo "========================================"
echo "FedAvg Experiment Completed"
echo "========================================"
