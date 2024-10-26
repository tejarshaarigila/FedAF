# server_fedaf.py

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from utils.utils import load_latest_model, compute_swd, ensure_directory_exists
import torch.optim as optim
from torchvision.utils import save_image
import logging

def train_model(model, train_loader, rc_tensor, num_classes, temperature, device, num_epochs, lambda_glob, logger):
    """
    Trains the model using the provided training data loader, including LGKM loss.

    Args:
        model (torch.nn.Module): The global model to train.
        train_loader (DataLoader): DataLoader for training data.
        rc_tensor (torch.Tensor): Aggregated class-wise soft labels from clients [10].
        num_classes (int): Number of classes.
        temperature (float): Temperature parameter for softmax scaling.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
        lambda_glob (float): Weight for the LGKM loss.
        logger (logging.Logger): Logger instance.
    """
    model.train()  # Set the model to training mode
    criterion_ce = nn.CrossEntropyLoss() 
    criterion_lgkm = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Ensure rc_tensor is a valid probability distribution
    epsilon = 1e-6
    rc_smooth = rc_tensor + epsilon
    rc_smooth = rc_smooth / rc_smooth.sum()

    current_lambda_glob = lambda_glob

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients at the start of each iteration

            outputs = model(inputs)  # Forward pass
            loss_ce = criterion_ce(outputs, labels)  # Compute Cross-Entropy loss

            # Softmax probabilities with temperature scaling
            log_probs = nn.functional.log_softmax(outputs / temperature, dim=1)  # Shape: [batch_size, 10]

            # Expand rc_smooth to match batch size
            Rc_target = rc_smooth.unsqueeze(0).repeat(labels.size(0), 1)  # Shape: [batch_size, 10]

            # Compute KL Divergence Loss
            loss_lgkm = criterion_lgkm(log_probs, Rc_target)

            # Total Loss
            loss = loss_ce + lambda_glob * loss_lgkm

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_epoch_loss = running_loss / len(train_loader)
        logger.info(f'Server: Epoch [{epoch +1}/{num_epochs}], Loss: {avg_epoch_loss:.4f} (CE: {loss_ce.item():.4f}, LGKM: {loss_lgkm.item():.4f}, Lambda: {current_lambda_glob:.4f})')

def evaluate_model(model, test_loader, device, logger):
    """
    Evaluates the model's performance on the test dataset.

    Args:
        model (torch.nn.Module): The global model to evaluate.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to evaluate on.
        logger (logging.Logger): Logger instance.

    Returns:
        float: Accuracy of the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f'Server: Model Accuracy on Test Data: {accuracy:.2f}%')
    return accuracy

def server_update(model_name, data, num_partitions, round_num, ipc, method, hratio, temperature, num_epochs, lambda_cdc, lambda_glob, device, logger):
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
        lambda_cdc (float): Weight for Client Data Condensation loss.
        lambda_glob (float): Weight for Global LGKM loss.
        device (str): Device to use for training ('cuda' or 'cpu').
        logger (logging.Logger): Logger instance.

    Returns:
        float or None: Accuracy of the updated model or None if skipped.
    """
    # Define paths and ensure directories exist
    data_path = '/home/t914a431/data'
    model_dir = os.path.join('/home/t914a431/models', data, model_name, str(num_partitions), str(hratio))
    ensure_directory_exists(model_dir)
    logger.info(f"Server: Model directory set to {model_dir}.")

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
        logger.error(f"Unsupported dataset: {data}")
        raise ValueError(f"Unsupported dataset: {data}")

    test_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=4)
    logger.info("Server: Test DataLoader created.")

    # Load aggregated class-wise soft labels Rc
    global_rc_path = os.path.join('/home/t914a431/logits', 'Global', f'Round{round_num}_Global_R.pt')
    if os.path.exists(global_rc_path):
        try:
            Rc_tensor = torch.load(global_rc_path, map_location=device)  # Shape: [10]
            Rc_tensor = Rc_tensor / Rc_tensor.sum()  # Normalize to sum to 1
            logger.info(f"Server: Loaded aggregated class-wise soft labels Rc from {global_rc_path}.")
        except Exception as e:
            logger.error(f"Server: Error loading aggregated Rc - {e}")
            Rc_tensor = torch.zeros(num_classes, device=device)  # Initialize with zeros if loading fails
    else:
        logger.warning(f"Server: Aggregated Rc not found at {global_rc_path}. Initializing with zeros.")
        Rc_tensor = torch.zeros(num_classes, device=device)  # Initialize with zeros

    # Aggregate synthetic data from clients
    all_images = []
    all_labels = []
    class_counts = torch.zeros(num_classes, device=device)

    logger.info("Server: Aggregating synthetic data from clients.")
    for client_id in range(num_partitions):
        synthetic_data_filename = os.path.join(
            '/home/t914a431/result',  # Adjusted based on main_fedaf.py
            f'Client_{client_id}',
            f'res_{method}_{data}_{model_name}_Client{client_id}_{ipc}ipc_Round{round_num}.pt'
        )

        if os.path.exists(synthetic_data_filename):
            try:
                data_dict = torch.load(synthetic_data_filename, map_location=device)
                if 'images' in data_dict and 'labels' in data_dict and data_dict['images'].size(0) > 0:
                    logger.info(f"Server: Loaded synthetic data from Client {client_id} at {synthetic_data_filename}.")
                    images, labels = data_dict['images'], data_dict['labels']
                    all_images.append(images)
                    all_labels.append(labels)

                    # Update class counts
                    for label in labels:
                        class_counts[label] += 1
                else:
                    logger.warning(f"Server: No valid synthetic data from Client {client_id}. Skipping.")
            except Exception as e:
                logger.error(f"Server: Error loading data from Client {client_id} - {e}")
        else:
            logger.warning(f"Server: No synthetic data found from Client {client_id} at {synthetic_data_filename}. Skipping.")

    # Check if any synthetic data was aggregated
    if not all_images:
        logger.error("Server: No synthetic data aggregated from any client. Skipping model update for this round.")
        return None  # Return None to indicate skipped round

    # Concatenate all synthetic data
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Proceed to create the DataLoader and train the model
    final_dataset = TensorDataset(all_images, all_labels)
    train_loader = DataLoader(final_dataset, batch_size=256, shuffle=True)
    logger.info("Server: Created DataLoader for training without data augmentation.")

    # Load the latest global model
    logger.info("Server: Loading the latest global model.")
    global_model = load_latest_model(
        model_dir=model_dir,
        model_name=model_name,
        channel=channel,
        num_classes=num_classes,
        im_size=im_size,
        device=device
    )
    logger.info("Server: Global model loaded.")

    # Compute T (class-wise averaged soft labels) from Rc_tensor
    logger.info("Server: Computing class-wise averaged soft labels T.")
    T_tensor = Rc_tensor.clone().detach()  # Shape: [10]
    logger.info(f"Server: Computed class-wise averaged soft labels T with shape {T_tensor.shape}.")

    # Train the global model
    logger.info("Server: Starting global model training.")
    train_model(
        model=global_model,
        train_loader=train_loader,
        rc_tensor=T_tensor,
        num_classes=num_classes,
        temperature=temperature,
        device=device,
        num_epochs=num_epochs,
        lambda_glob=lambda_glob,
        logger=logger
    )
    logger.info("Server: Global model training completed.")

    # Evaluate the updated global model
    logger.info("Server: Evaluating the updated global model.")
    accuracy = evaluate_model(
        model=global_model,
        test_loader=test_loader,
        device=device,
        logger=logger
    )
    logger.info(f"Server: Updated global model accuracy: {accuracy:.2f}%.")

    # Save the updated global model
    model_path = os.path.join(model_dir, f"fedaf_global_model_{round_num}.pth")
    try:
        ensure_directory_exists(os.path.dirname(model_path))
        torch.save(global_model.state_dict(), model_path)
        logger.info(f"Server: Updated global model saved to {model_path}.")
    except Exception as e:
        logger.error(f"Server: Error saving the global model to {model_path} - {e}")

    return accuracy  # Return accuracy for tracking
