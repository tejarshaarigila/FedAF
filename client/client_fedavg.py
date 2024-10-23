# client_fedavg.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import get_network  # Updated import
import logging
import os

class Client:
    def __init__(self, client_id, train_data, args):
        """
        Initializes a federated learning client.

        Args:
            client_id (int): Unique identifier for the client.
            train_data (Subset): Subset of the dataset assigned to the client.
            args (Namespace): Parsed arguments containing configurations.
        """
        self.client_id = client_id
        self.args = args
        self.device = args.device
        self.model = get_network(
            model_name=args.model,
            channel=args.channel,
            num_classes=args.num_classes,
            im_size=args.im_size,
            device=self.device
        )
        self.train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0  # Ensure num_workers=0 for multiprocessing
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

        # Set up a dedicated logger for the client
        self.logger = logging.getLogger(f'FedAvg.Client{self.client_id}')
        self.setup_logger()

    def setup_logger(self):
        """
        Sets up the logger for the client to log to a separate file.
        """
        log_dir = "/home/t914a431/log/client_logs/"
        os.makedirs(log_dir, exist_ok=True)
        client_log_file = os.path.join(log_dir, f'client_{self.client_id}.log')

        # Prevent adding multiple handlers if already added
        if not self.logger.handlers:
            file_handler = logging.FileHandler(client_log_file)
            console_handler = logging.StreamHandler()

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def set_model(self, model_state_dict):
        """
        Sets the client's model state to the global model state.

        Args:
            model_state_dict (dict): State dictionary of the global model.
        """
        self.model.load_state_dict(model_state_dict)
        self.logger.info("Client %d: Model state updated from global model.", self.client_id)

    def train(self):
        """
        Trains the local model on the client's data.

        Returns:
            dict: State dictionary of the trained model.
        """
        self.logger.info("Client %d: Starting local training.", self.client_id)
        self.model.train()
        for epoch in range(1, self.args.local_epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            avg_loss = epoch_loss / len(self.train_loader.dataset)
            accuracy = 100 * correct / total
            self.logger.info("Client %d: Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%",
                             self.client_id, epoch, self.args.local_epochs, avg_loss, accuracy)

        self.logger.info("Client %d: Local training completed.", self.client_id)
        return self.model.state_dict()
