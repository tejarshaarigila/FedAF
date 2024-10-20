#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8GB
#SBATCH -t 30:00:00 
#SBATCH -J cifar10
#SBATCH -o slurm-%j.out

# Specify the paths to your Python files
PYTHON_FILE_1="main_fedavg.py"
PYTHON_FILE_2="main_fedaf.py"
PYTHON_FILE_3="main_plot.py"

# Run the first Python file
echo "Running first Python file: $PYTHON_FILE_1"
python3 $PYTHON_FILE_1

# Check if the first script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: First script failed."
    exit 1
fi

# Run the second Python file
echo "Running second Python file: $PYTHON_FILE_2"
python3 $PYTHON_FILE_2

# Check if the second script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: Second script failed."
    exit 1
fi

# Run the third Python file
echo "Running third Python file: $PYTHON_FILE_3"
python3 $PYTHON_FILE_3

# Check if the third script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: Third script failed."
    exit 1
fi

echo "All three scripts ran successfully."
