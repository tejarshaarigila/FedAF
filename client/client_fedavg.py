# client_fedavg.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils_fedavg import get_network
import logging

# Configure logging
logger = logging.getLogger(__name__)

class Client:
    def __init__(self, client_id, train_data, args):
        self.client_id = client_id
        self.args = args
        self.train_data = train_data  # Already a Dataset or Subset
        self.train_loader = DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True)
        self.local_model = self.initialize_model()
        logger.info("Client %d initialized with model: %s", client_id, args.model)

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
        
        logger.info("Local model initialized for client %d.", self.client_id)
        return model

    def set_model(self, global_model):
        self.local_model.load_state_dict(global_model)
        logger.info("Client %d set the global model.", self.client_id)

    def train(self):
        """Train the client's model on local data."""
        self.local_model.train()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.args.local_epochs):
            epoch_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                outputs = self.local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            logger.info("Client %d completed epoch %d with average loss: %.4f", self.client_id, epoch + 1, avg_loss)

        return self.local_model.state_dict()
