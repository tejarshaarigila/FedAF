#!/bin/bash

#SBATCH -p intel
#SBATCH -N 1              # Number of nodes
#SBATCH -n 1              # Number of tasks
#SBATCH -c 10             # Total number of CPUs
#SBATCH --mem=32G     
#SBATCH -t 48:00:00       
#SBATCH -J federated_exp
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=1  
export MKL_NUM_THREADS=1 

# Python scripts
PYTHON_FILE_PARTITION="utils/generate_partitions.py"
PYTHON_FILE_FEDAVG="main_fedavg.py"
PYTHON_FILE_FEDAF="main_fedaf.py"
PYTHON_FILE_PLOT="main_plot.py"

# parameters
DATASET="CIFAR10"
ALPHA_DIRICHLET=0.1
HONESTY_RATIO=1.0
MODEL="ConvNet"
SAVE_DIR="/home/t914a431/results/"
PARTITION_BASE_DIR="/home/t914a431/partitions_per_round"
NUM_USERS_LIST=(5 10 15 20)

echo "========================================"
echo "Starting Federated Learning Experiments"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "========================================"

echo "Cleaning existing data directories..."
rm -rf /home/t914a431/data/*
mkdir -p /home/t914a431/data

echo "Pre-downloading the CIFAR-10 dataset..."
srun -n 1 -c 1 python3 -c "from torchvision import datasets; datasets.CIFAR10(root='/home/t914a431/data', train=True, download=True); datasets.CIFAR10(root='/home/t914a431/data', train=False, download=True)"

if [ $? -ne 0 ]; then
    echo "Error: Dataset download failed."
    exit 1
fi
echo "Dataset downloaded successfully."

for NUM_USERS in "${NUM_USERS_LIST[@]}"; do
    echo "========================================"
    echo "Starting experiments for NUM_USERS=${NUM_USERS}"
    echo "========================================"

    MODEL_BASE_DIR="/home/t914a431/models/${DATASET}/${MODEL}/${NUM_USERS}/${HONESTY_RATIO}/"
    PARTITION_DIR="${PARTITION_BASE_DIR}/${DATASET}/${MODEL}/${NUM_USERS}/${HONESTY_RATIO}/"

    mkdir -p "$MODEL_BASE_DIR"
    mkdir -p "$PARTITION_DIR"

    echo "----------------------------------------"
    echo "Step 1: Generating Data Partitions"
    echo "----------------------------------------"

    echo "Running Partition Generation: $PYTHON_FILE_PARTITION with ${DATASET}, ${NUM_USERS} clients, alpha=${ALPHA_DIRICHLET}"
    srun -n 1 -c 10 python3 $PYTHON_FILE_PARTITION \
        --dataset $DATASET \
        --num_clients $NUM_USERS \
        --num_rounds 20 \
        --alpha $ALPHA_DIRICHLET \
        --honesty_ratio $HONESTY_RATIO \
        --seed 42

    status_partition=$?
    if [ $status_partition -ne 0 ]; then
        echo "Error: Partition generation failed for NUM_USERS=${NUM_USERS}."
        exit 1
    fi
    echo "Partition generation completed successfully for NUM_USERS=${NUM_USERS}."

    echo "----------------------------------------"
    echo "Step 2: Running FedAF"
    echo "----------------------------------------"

    echo "Running FedAF: $PYTHON_FILE_FEDAF with ${DATASET}, ${NUM_USERS} clients, alpha=${ALPHA_DIRICHLET}, honesty_ratio=${HONESTY_RATIO}"
    srun -n 1 -c 10 python3 $PYTHON_FILE_FEDAF \
        --dataset $DATASET \
        --model $MODEL \
        --num_clients $NUM_USERS \
        --alpha $ALPHA_DIRICHLET \
        --honesty_ratio $HONESTY_RATIO \
        --partition_dir "$PARTITION_DIR"

    status_fedaf=$?
    if [ $status_fedaf -ne 0 ]; then
        echo "Error: FedAF script failed for NUM_USERS=${NUM_USERS}."
        exit 1
    fi
    echo "FedAF script completed successfully for NUM_USERS=${NUM_USERS}."

    echo "----------------------------------------"
    echo "Step 3: Running FedAvg"
    echo "----------------------------------------"

    echo "Running FedAvg: $PYTHON_FILE_FEDAVG with ${DATASET}, ${NUM_USERS} clients, alpha=${ALPHA_DIRICHLET}, honesty_ratio=${HONESTY_RATIO}"
    srun -n 1 -c 10 python3 $PYTHON_FILE_FEDAVG \
        --dataset $DATASET \
        --model $MODEL \
        --num_clients $NUM_USERS \
        --alpha $ALPHA_DIRICHLET \
        --honesty_ratio $HONESTY_RATIO \
        --partition_dir "$PARTITION_DIR" \
        --save_dir "$MODEL_BASE_DIR" \
        --log_dir "/home/t914a431/log/"

    status_fedavg=$?
    if [ $status_fedavg -ne 0 ]; then
        echo "Error: FedAvg script failed for NUM_USERS=${NUM_USERS}."
        exit 1
    fi
    echo "FedAvg script completed successfully for NUM_USERS=${NUM_USERS}."

    echo "----------------------------------------"
    echo "Step 4: Running Plotting Script"
    echo "----------------------------------------"

    echo "Running Plot: $PYTHON_FILE_PLOT for ${DATASET}, ${NUM_USERS} clients, alpha=${ALPHA_DIRICHLET}"
    srun -n 1 -c 10 python3 $PYTHON_FILE_PLOT \
        --dataset $DATASET \
        --model $MODEL \
        --num_users $NUM_USERS \
        --alpha_dirichlet $ALPHA_DIRICHLET \
        --honesty_ratio $HONESTY_RATIO \
        --model_base_dir "$MODEL_BASE_DIR" \
        --save_dir "$SAVE_DIR"

    status_plot=$?
    if [ $status_plot -ne 0 ]; then
        echo "Error: Plot script failed for NUM_USERS=${NUM_USERS}."
        exit 1
    fi
    echo "Plot script completed successfully for NUM_USERS=${NUM_USERS}."

    echo "========================================"
    echo "Completed experiments for NUM_USERS=${NUM_USERS}"
    echo "========================================"
done

echo "All experiments for $DATASET completed successfully."
