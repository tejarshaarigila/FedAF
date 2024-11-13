#!/bin/bash

#SBATCH -p intel
#SBATCH -N 1             
#SBATCH -n 1              
#SBATCH -c 10             
#SBATCH --mem=24G     
#SBATCH -t 00:00:00       
#SBATCH -J fedaf_exp
#SBATCH -o slurm-fedaf-%j.out


PYTHON_FILE_FEDAF="main_fedaf.py"
DATASET="MNIST"
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
echo "Starting FedAF Experiment"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Number of Clients: $NUM_USERS"
echo "Alpha (Dirichlet): $ALPHA_DIRICHLET"
echo "Honesty Ratio: $HONESTY_RATIO"
echo "========================================"

if [ ! -d "${DATA_PATH}/${DATASET}" ]; then
    echo "Pre-downloading the dataset..."
    srun -n 1 -c 1 python3 -c "from torchvision import datasets; datasets.${DATASET}(root='${DATA_PATH}', train=True, download=True); datasets.${DATASET}(root='${DATA_PATH}', train=False, download=True)"
fi

echo "----------------------------------------"
echo "Running FedAF"
echo "----------------------------------------"

srun -n 1 -c 10 python3 $PYTHON_FILE_FEDAF

if [ $? -ne 0 ]; then
    echo "Error: FedAF script failed."
    exit 1
fi
echo "FedAF script completed successfully."

echo "========================================"
echo "FedAF Experiment Completed"
echo "========================================"
