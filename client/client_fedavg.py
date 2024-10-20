# client_fedavg.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from utils.utils_fedavg import get_network
import logging

logger = logging.getLogger(__name__)

class Client:
    def __init__(self, client_id, train_data, args):
        self.client_id = client_id
        self.args = args
        self.device = args.device
        self.model = get_network(
            args.model,
            args.channel,
            args.num_classes,
            args.im_size,
            device=self.device
        )
        self.train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0  # Ensure num_workers=0
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

    def set_model(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)

    def train(self):
        self.model.train()
        for epoch in range(self.args.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        # Log when client completes local training
        logger.info(f"Client {self.client_id} has completed local training.")
        return self.model.state_dict()
