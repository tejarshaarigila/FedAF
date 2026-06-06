# server_fedavg.py

import os
import torch
import logging
from utils.utils_fedavg import get_network

# Configure logging
logger = logging.getLogger(__name__)

class Server:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_clients = cfg.num_clients
        self.global_model = self.initialize_model()
        logger.info("Server initialized with model: %s for dataset: %s", cfg.model, cfg.dataset)

    def initialize_model(self):
        model = get_network(
            model=self.cfg.model,
            channel=self.cfg.channel,
            num_classes=self.cfg.num_classes,
            im_size=self.cfg.im_size,
            device=self.cfg.device,
        ).to(self.cfg.device)

        logger.info("Global model initialized.")
        return model

    def get_global_model(self):
        return self.global_model.state_dict()

    def model_path(self, round_num):
        """Checkpoint path for a given round (lives in the shared cfg.model_dir)."""
        return os.path.join(self.cfg.model_dir, f"fedavg_global_model_{round_num}.pth")

    def load_round(self, round_num):
        """Restore the global model from a saved round checkpoint (for resume)."""
        path = self.model_path(round_num)
        self.global_model.load_state_dict(torch.load(path, map_location=self.cfg.device))
        logger.info("Resumed FedAvg global model from %s", path)

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
                images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        # Save the global model after evaluation (shared, config-driven location)
        model_save_path = self.model_path(round_num)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.global_model.state_dict(), model_save_path)

        logger.info("Global Model Accuracy: %.2f%%", accuracy)
        logger.info("Global model saved at: %s", model_save_path)

        return accuracy
