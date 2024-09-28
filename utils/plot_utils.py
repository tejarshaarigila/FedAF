# plot_utils.py

import matplotlib.pyplot as plt
import os


def plot_accuracies(test_accuracies, model_name, dataset, alpha, num_clients, save_dir="visualizations"):
    """
    Plot and save the test accuracies over the federated learning rounds.

    Args:
        test_accuracies (list of float): List of test accuracies after each round.
        model_name (str): Name of the model used (e.g., 'ConvNet', 'ResNet').
        dataset (str): Name of the dataset (e.g., 'CIFAR10', 'MNIST', 'CelebA').
        alpha (float): Alpha parameter for the Dirichlet distribution.
        num_clients (int): Number of clients in federated learning.
        save_dir (str): Directory where the plots will be saved.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate the filename
    filename = f"vis_{model_name}_{dataset}_alpha{alpha}_u{num_clients}.png"
    save_path = os.path.join(save_dir, filename)

    # Plot test accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(test_accuracies, marker='o', linestyle='-')
    plt.title(f"Test Accuracy over Rounds\nModel: {model_name}, Dataset: {dataset}, Alpha: {alpha}, Clients: {num_clients}")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved test accuracy plot to {save_path}")
