# server_fedavg.py

import os
import torch
import logging
from utils.utils_fedavg import get_network

# Configure logging
logger = logging.getLogger(__name__)

class Server:
    def __init__(self, args):
        self.args = args
        self.num_clients = args.num_clients
        self.global_model = self.initialize_model()
        logger.info("Server initialized with model: %s for dataset: %s", args.model, args.dataset)

    def initialize_model(self):
        im_size = self.args.im_size
        channel = self.args.channel
        num_classes = self.args.num_classes

        model = get_network(
            model=self.args.model,
            channel=channel,
            num_classes=num_classes,
            im_size=im_size
        ).to(self.args.device)
        
        logger.info("Global model initialized.")
        return model

    def get_global_model(self):
        return self.global_model.state_dict()

    def aggregate(self, client_models, data_sizes):
        """Aggregates client models using FedAvg with weighting by data sizes."""
        logger.info("Aggregating client models with data size weighting.")
        global_dict = self.global_model.state_dict()
        total_samples = sum(data_sizes)
        averaged_params = {key: torch.zeros_like(param) for key, param in global_dict.items()}

        # Sum the weighted parameters from each client
        for client_model, data_size in zip(client_models, data_sizes):
            weight = data_size / total_samples
            for key in global_dict.keys():
                averaged_params[key] += client_model[key] * weight

        # Update the global model parameters
        self.global_model.load_state_dict(averaged_params)
        logger.info("Model aggregation completed.")

    def evaluate(self, test_loader, round_num):
        """Evaluates the global model on test data and saves the model."""
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        
        # Save the global model after evaluation
        model_save_path = f"/home/t914a431/models/{self.args.dataset}/{self.args.model}/{self.args.num_clients}/{str(int(self.args.honesty_ratio))}/fedavg_global_model_{round_num}.pth"
        
        # Extract the directory path from the model save path
        model_save_dir = os.path.dirname(model_save_path)

        # Check if the directory exists, if not, create it
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        torch.save(self.global_model.state_dict(), model_save_path)
        
        logger.info("Global Model Accuracy: %.2f%%", accuracy)
        logger.info("Global model saved at: %s", model_save_path)
        
        return accuracy
