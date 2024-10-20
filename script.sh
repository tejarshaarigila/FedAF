#!/bin/bash

#SBATCH -p intel
#SBATCH -N 1          # Number of nodes
#SBATCH -n 2          # Number of tasks (we will run 2 tasks in parallel)
#SBATCH -c 20          # CPUs per task
#SBATCH --mem=32G
#SBATCH -t 30:00:00
#SBATCH -J cifar10
#SBATCH -o slurm-%j.out

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

# Run the first Python file using srun in the background
echo "Running first Python file: $PYTHON_FILE_1"
srun -n 1 -c 10 --mem=16G python3 $PYTHON_FILE_1 &
pid1=$!

# Run the second Python file using srun in the background
echo "Running second Python file: $PYTHON_FILE_2"
srun -n 1 -c 10 --mem=16G python3 $PYTHON_FILE_2 &
pid2=$!

# Wait for both srun commands to complete
wait $pid1
status1=$?
wait $pid2
status2=$?

# Check if the first script ran successfully
if [ $status1 -ne 0 ]; then
    echo "Error: First script failed."
    exit 1
fi

# Check if the second script ran successfully
if [ $status2 -ne 0 ]; then
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
