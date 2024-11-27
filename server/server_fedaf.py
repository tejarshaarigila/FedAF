# server_fedaf.py

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from utils.utils_fedaf import load_latest_model, ensure_directory_exists
import torch.optim as optim

def train_model(model, train_loader, Rc_tensor, num_classes, lambda_glob, temperature, device, num_epochs):
    """
    Trains the model using the provided training data loader, including LGKM loss.

    Args:
        model (torch.nn.Module): The global model to train.
        train_loader (DataLoader): DataLoader for training data.
        Rc_tensor (torch.Tensor): Aggregated class-wise soft labels from clients (shape: [num_classes,]).
        num_classes (int): Number of classes.
        lambda_glob (float): Regularization parameter for LGKM loss.
        temperature (float): Temperature for softmax scaling.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
    """
    model.to(device)
    model.train()
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    epsilon = 1e-8
    Rc_smooth = Rc_tensor + epsilon  # [num_classes,]

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_ce_loss = 0.0
        running_lgkm_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
        
            outputs = model(inputs)  # [batch_size, num_classes]
            loss_ce = criterion_ce(outputs, labels)
        
            soft_labels = nn.functional.softmax(outputs / temperature, dim=1)  # [batch_size, num_classes]
        
            # Initialize T
            T = torch.zeros(num_classes, device=device)
        
            # Aggregate soft labels class-wise
            for c in range(num_classes):
                mask = labels == c  # [batch_size,]
                if mask.sum() > 0:
                    T[c] = soft_labels[mask].mean(dim=0)[c]
                else:
                    T[c] = 1.0 / num_classes

            T = T + epsilon
        
            # LGKM loss using KL Divergence
            Rc_normalized = Rc_smooth / Rc_smooth.sum()
            T_normalized = T / T.sum()
        
            kl_div1 = nn.functional.kl_div(T_normalized.log(), Rc_normalized, reduction='mean')
            kl_div2 = nn.functional.kl_div(Rc_normalized.log(), T_normalized, reduction='mean')
            loss_lgkm = (kl_div1 + kl_div2) / 2
        
            # Combine losses
            combined_loss = loss_ce + lambda_glob * loss_lgkm
        
            combined_loss.backward()
            optimizer.step()
        
            running_loss += combined_loss.item()
            running_ce_loss += loss_ce.item()
            running_lgkm_loss += loss_lgkm.item()
        
        avg_loss = running_loss / len(train_loader)
        avg_ce_loss = running_ce_loss / len(train_loader)
        avg_lgkm_loss = running_lgkm_loss / len(train_loader)
        
        print(f'Server: Epoch {epoch + 1}, Total Loss: {avg_loss:.4f}, CE Loss: {avg_ce_loss:.4f}, LGKM Loss: {avg_lgkm_loss:.4f}')
        running_loss = 0.0

def evaluate_model(model, test_loader, device):
    """
    Evaluates the model's performance on the test dataset.

    Args:
        model (torch.nn.Module): The global model to evaluate.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to evaluate on.
    """
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():  # No gradients needed
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on test images: {accuracy:.2f}%')

def server_update(model_name, data, num_partitions, round_num, lambda_glob, ipc, method, hratio, temperature, num_epochs, device="cpu"):
    """
    Aggregates synthetic data from all clients, updates the global model, evaluates it,
    and computes aggregated logits to send back to clients.

    Args:
        model_name (str): Model architecture (e.g., 'ConvNet').
        data (str): Dataset name ('CIFAR10' or 'MNIST').
        num_partitions (int): Number of client partitions.
        round_num (int): Current communication round number.
        ipc (int): Instances per class.
        method (str): Method used, e.g., 'fedaf'.
        hratio (float): Honesty ratio for client honesty.
        temperature (float): Temperature for softmax scaling.
        num_epochs (int): Number of epochs for training.
        device (str): Device to use for training ('cuda' or 'cpu').
    """
    # Define paths and ensure directories exist
    data_path = '/home/data'
    model_dir = os.path.join('/home/models', data, model_name, str(num_partitions), str(hratio))
    ensure_directory_exists(model_dir)

    # Load test dataset with necessary transformations
    if data == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    elif data == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {data}")

    test_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=4)

    # Load aggregated class-wise soft labels Rc
    global_probs_path = os.path.join('/home/logits', 'Global', f'Round{round_num}_Global_Rc.pt')
    if os.path.exists(global_probs_path):
        Rc = torch.load(global_probs_path, map_location=device)
        # Ensure Rc is of shape [num_classes,]
        if Rc.dim() == 1 and Rc.size(0) == num_classes:
            print(f"Server: Loaded aggregated class-wise soft labels R(c) with shape {Rc.shape}.")
        else:
            raise ValueError(f"Rc has incorrect shape: {Rc.shape}. Expected [{num_classes}].")
    else:
        print("Server: No aggregated class-wise soft labels found. Initializing R(c) with zeros.")
        Rc = torch.zeros(num_classes, device=device)

    all_images = []
    all_labels = []

    # Aggregate synthetic data from all clients
    print("Server: Aggregating synthetic data from clients.")
    for client_id in range(num_partitions):
        synthetic_data_filename = os.path.join(
            '/home',
            # '',
            'result',
            f'Client_{client_id}',
            f'res_{method}_{data}_{model_name}_Client{client_id}_{ipc}ipc_Round{round_num}.pt'
        )

        if os.path.exists(synthetic_data_filename):
            try:
                data_dict = torch.load(synthetic_data_filename, map_location=device)
                if 'images' in data_dict and 'labels' in data_dict and data_dict['images'].size(0) > 0:
                    print(f"Server: Loaded synthetic data from Client {client_id}.")
                    images, labels = data_dict['images'], data_dict['labels']
                    all_images.append(images)
                    all_labels.append(labels)
                else:
                    print(f"Server: No valid synthetic data from Client {client_id}. Skipping.")
            except Exception as e:
                print(f"Server: Error loading data from Client {client_id} - {e}")
        else:
            print(f"Server: No synthetic data found for Client {client_id} at {synthetic_data_filename}. Skipping.")

    if not all_images:
        print("Server: No synthetic data aggregated from clients. Skipping model update.")
        return

    # Concatenate all synthetic data
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Create training dataset and loader
    final_dataset = TensorDataset(all_images, all_labels)
    train_loader = DataLoader(final_dataset, batch_size=256, shuffle=True, num_workers=4)

    # Load the latest global model
    print("Server: Loading the latest global model.")
    net = load_latest_model(model_dir, model_name, channel, num_classes, im_size, device)
    net.train()

    Rc_tensor = Rc.to(device)  # Shape: [num_classes,]

    # Train the global model
    print("Server: Starting global model training.")
    train_model(net, train_loader, Rc_tensor, num_classes, lambda_glob, temperature, device, num_epochs=num_epochs)

    # Evaluate the updated global model
    print("Server: Evaluating the updated global model.")
    evaluate_model(net, test_loader, device)

    # Save the updated global model
    model_path = os.path.join(model_dir, f"fedaf_global_model_{round_num}.pth")
    try:
        ensure_directory_exists(os.path.dirname(model_path))
        torch.save(net.state_dict(), model_path)
        print(f"Server: Global model updated and saved to {model_path}.")
    except Exception as e:
        print(f"Server: Error saving the global model - {e}")
