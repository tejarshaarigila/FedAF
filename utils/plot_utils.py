# plot_utils.py

import matplotlib.pyplot as plt
import os
import logging

def setup_plot_utils_logger():
    """
    Sets up the logger for plot_utils to log to a separate file.
    Ensures that the log directory exists before creating the log file.
    """
    log_dir = "/home/t914a431/log/"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists

    logger = logging.getLogger('FedAvg.PlotUtils')
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if already added
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'plot_utils.log'))
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_plot_utils_logger()

def plot_accuracies(test_accuracies, model_name, dataset, alpha, num_clients):
    """
    Plots and saves the test accuracies over communication rounds.

    Args:
        test_accuracies (list): List of test accuracies per round.
        model_name (str): Name of the model architecture.
        dataset (str): Dataset name.
        alpha (float): Dirichlet distribution parameter.
        num_clients (int): Number of clients.
    """
    try:
        rounds = range(1, len(test_accuracies) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, test_accuracies, marker='o', linestyle='-')
        plt.title(f'FedAvg Test Accuracy: {model_name} on {dataset}\nAlpha={alpha}, Clients={num_clients}')
        plt.xlabel('Communication Round')
        plt.ylabel('Test Accuracy (%)')
        plt.xticks(rounds)
        plt.grid(True)

        # Save the plot
        plot_dir = "/home/t914a431/plots/"
        os.makedirs(plot_dir, exist_ok=True)  # Ensure the plot directory exists
        plot_filename = f'fedavg_{model_name}_{dataset}_alpha{alpha}_clients{num_clients}.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Test accuracy plot saved to {plot_path}.")
    except Exception as e:
        logger.error(f"Error while plotting accuracies: {e}")
