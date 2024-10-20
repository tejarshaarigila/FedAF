#!/bin/bash

#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1          # Number of tasks (processes)
#SBATCH -c 10         # CPUs per task
#SBATCH --mem=16G
#SBATCH -t 30:00:00
#SBATCH -J cifar10
#SBATCH -o slurm-%j.out

# Load necessary modules (if any)
# module load python/3.9

# Set environment variables
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

# Specify the paths to your Python files
PYTHON_FILE_1="main_fedavg.py"
PYTHON_FILE_2="main_fedaf.py"
PYTHON_FILE_3="main_plot.py"

# Clean existing data directory to avoid corrupted data
echo "Cleaning existing data directories..."
rm -rf data/*

# Pre-download the CIFAR-10 dataset
echo "Pre-downloading the CIFAR-10 dataset..."
python3 -c "from torchvision import datasets; datasets.CIFAR10(root='data', train=True, download=True); datasets.CIFAR10(root='data', train=False, download=True)"

# Check if the dataset was downloaded successfully
if [ $? -ne 0 ]; then
    echo "Error: Dataset download failed."
    exit 1
fi

# Run the first Python file using srun
echo "Running first Python file: $PYTHON_FILE_1"
srun -n 1 -c 10 --mem=16G python3 $PYTHON_FILE_1

# Check if the first script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: First script failed."
    exit 1
fi

# Run the second Python file using srun
echo "Running second Python file: $PYTHON_FILE_2"
srun -n 1 -c 10 --mem=16G python3 $PYTHON_FILE_2

# Check if the second script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: Second script failed."
    exit 1
fi

# Run the third Python file using srun
echo "Running third Python file: $PYTHON_FILE_3"
srun -n 1 -c 10 --mem=16G python3 $PYTHON_FILE_3

# Check if the third script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: Third script failed."
    exit 1
fi

echo "All scripts ran successfully."
