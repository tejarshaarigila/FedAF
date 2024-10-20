# server_fedavg.py

import torch
import logging
from utils.utils_fedavg import get_network

logger = logging.getLogger(__name__)

class Server:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model = get_network(
            args.model,
            args.channel,
            args.num_classes,
            args.im_size,
            device=self.device
        )

    def get_global_model_state(self):
        return self.model.state_dict()

    def aggregate(self, client_models, client_sizes):
        total_size = sum(client_sizes)
        new_state_dict = {}
        for key in self.model.state_dict().keys():
            new_state_dict[key] = torch.zeros_like(self.model.state_dict()[key])
            for client_model, client_size in zip(client_models, client_sizes):
                new_state_dict[key] += client_model[key] * (client_size / total_size)
        self.model.load_state_dict(new_state_dict)

    def evaluate(self, test_loader, round_num):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        logger.info(f'Round {round_num} - Test Accuracy: {accuracy:.2f}%')
        return accuracy

    def save_model(self, round_num):
        model_path = f'global_model_round_{round_num}.pth'
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Global model saved to {model_path}")
