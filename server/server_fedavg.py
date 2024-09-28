# server_fedavg.py

import os
import torch
import logging
from utils.networks import ConvNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Server:
    def __init__(self, args):
        self.args = args
        self.global_model = self.initialize_model()
        logger.info("Server initialized with model: %s for dataset: %s", args.model, args.dataset)

    def initialize_model(self):
        if self.args.dataset == 'CIFAR10':
            model = ConvNet(channel=3, num_classes=10, net_width=32, net_depth=2, net_act='relu', net_norm='batchnorm', net_pooling='maxpooling').to(self.args.device)
        elif self.args.dataset == 'MNIST':
            model = ConvNet(channel=1, num_classes=10, net_width=16, net_depth=2, net_act='relu', net_norm='batchnorm', net_pooling='maxpooling').to(self.args.device)
        elif self.args.dataset == 'CelebA':
            model = ConvNet(channel=3, num_classes=2, net_width=32, net_depth=2, net_act='relu', net_norm='batchnorm', net_pooling='maxpooling').to(self.args.device)
        else:
            raise ValueError("Unsupported dataset.")
        logger.info("Global model initialized.")
        return model

    def get_global_model(self):
        return self.global_model.state_dict()

    def aggregate(self, client_models):
        """Aggregates client models using FedAvg."""
        logger.info("Aggregating client models.")
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([client_model[key].float() for client_model in client_models], 0).mean(0)
        self.global_model.load_state_dict(global_dict)
        logger.info("Model aggregation completed.")

    def evaluate(self, test_loader, r):
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
        model_save_path = f"models/{self.args.dataset}/{self.args.model}/{self.num_classes}/{self.args.honesty_ratio}/{self.args.num_clients}/fedavg_global_model_{r}.pth"
        
        # Extract the directory path from the model save path
        model_save_dir = os.path.dirname(model_save_path)

        # Check if the directory exists, if not, create it
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        torch.save(self.global_model.state_dict(), model_save_path)
        
        logger.info("Global Model Accuracy: %.2f%%", accuracy)
        logger.info("Global model saved at: %s", model_save_path)
        
        return accuracy
