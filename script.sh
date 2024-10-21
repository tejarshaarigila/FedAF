#!/bin/bash

#SBATCH -p intel
#SBATCH -N 1          # Number of nodes
#SBATCH -n 2          # Number of tasks
#SBATCH -c 10         # CPUs per task
#SBATCH --mem=32G
#SBATCH -t 30:00:00
#SBATCH -J cifar10
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5

PYTHON_FILE_FEDAVG="main_fedavg.py"
PYTHON_FILE_FEDAF="main_fedaf.py"
PYTHON_FILE_PLOT="main_plot.py"

# parameters
DATASET="CIFAR10"
ALPHA_DIRICHLET=0.1
HONESTY_RATIO=1
MODEL="ConvNet"
SAVE_DIR="/home/t914a431/results/"

NUM_USERS_LIST=(5 10 15 20)

echo "Cleaning existing data directories..."
rm -rf data/*

echo "Pre-downloading the CIFAR-10 dataset..."
python3 -c "from torchvision import datasets; datasets.CIFAR10(root='data', train=True, download=True); datasets.CIFAR10(root='data', train=False, download=True)"

if [ $? -ne 0 ]; then
    echo "Error: Dataset download failed."
    exit 1
fi

for NUM_USERS in "${NUM_USERS_LIST[@]}"; do
    echo "========================================"
    echo "Starting experiments for NUM_USERS=${NUM_USERS}"
    echo "========================================"

    MODEL_BASE_DIR="./models/${DATASET}/${MODEL}/${NUM_USERS}/${HONESTY_RATIO}/"

    echo "Running FedAF: $PYTHON_FILE_FEDAF with ${DATASET}, ${NUM_USERS} clients, alpha=${ALPHA_DIRICHLET}, honesty_ratio=${HONESTY_RATIO}"
    srun -n 1 -c 5 --mem=16G python3 $PYTHON_FILE_FEDAF \
        --dataset $DATASET \
        --model $MODEL \
        --num_partitions $NUM_USERS \
        --alpha $ALPHA_DIRICHLET \
        --honesty_ratio $HONESTY_RATIO &
    pid_fedaf=$!

    echo "Running FedAvg: $PYTHON_FILE_FEDAVG with ${DATASET}, ${NUM_USERS} clients, alpha=${ALPHA_DIRICHLET}, honesty_ratio=${HONESTY_RATIO}"
    srun -n 1 -c 5 --mem=16G python3 $PYTHON_FILE_FEDAVG \
        --dataset $DATASET \
        --model $MODEL \
        --num_clients $NUM_USERS \
        --alpha $ALPHA_DIRICHLET \
        --honesty_ratio $HONESTY_RATIO &
    pid_fedavg=$!

    wait $pid_fedaf
    status_fedaf=$?
    wait $pid_fedavg
    status_fedavg=$?

    if [ $status_fedaf -ne 0 ]; then
        echo "Error: FedAF script failed for NUM_USERS=${NUM_USERS}."
        exit 1
    fi

    if [ $status_fedavg -ne 0 ]; then
        echo "Error: FedAvg script failed for NUM_USERS=${NUM_USERS}."
        exit 1
    fi

    echo "FedAF and FedAvg scripts completed successfully for NUM_USERS=${NUM_USERS}."

    echo "Running Plot: $PYTHON_FILE_PLOT for ${DATASET}, ${NUM_USERS} clients, alpha=${ALPHA_DIRICHLET}"
    srun -n 1 -c 10 --mem=16G python3 $PYTHON_FILE_PLOT \
        --dataset $DATASET \
        --model $MODEL \
        --num_users $NUM_USERS \
        --alpha_dirichlet $ALPHA_DIRICHLET \
        --honesty_ratio $HONESTY_RATIO \
        --save_dir $SAVE_DIR

    if [ $? -ne 0 ]; then
        echo "Error: Plot script failed for NUM_USERS=${NUM_USERS}."
        exit 1
    fi

    echo "Plot script completed successfully for NUM_USERS=${NUM_USERS}."
    echo "========================================"
    echo "Completed experiments for NUM_USERS=${NUM_USERS}"
    echo "========================================"
done

echo "All experiments for CIFAR10 completed successfully."
