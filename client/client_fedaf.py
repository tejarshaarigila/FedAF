# client_fedaf.py

import os
import copy
import numpy as np
import torch
from torchvision.utils import save_image
from torch import optim
from torch.utils.data import DataLoader, Subset
from utils.utils_fedaf import (
    load_latest_model,
    calculate_logits_labels,
    compute_swd,
)
import logging

# Configure logging to save log file in the specified directory
log_dir = "/home/t914a431/log/"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(log_dir, 'fedaf_client.log'),  # Log file path
    filemode='w'
)
logger = logging.getLogger(__name__)


class Client:
    def __init__(self, client_id: int, data_partition: Subset, args_dict: dict):
        """
        Initializes a federated learning client.

        Args:
            client_id (int): Unique identifier for the client.
            data_partition (Subset): Subset of the dataset assigned to the client.
            args_dict (dict): Dictionary containing hyperparameters and configurations.
        """
        self.client_id = client_id
        self.data_partition = data_partition
        self.args = args_dict

        # Required configurations
        required_args = [
            'device', 'dataset', 'model', 'num_classes',
            'channel', 'im_size', 'ipc', 'temperature',
            'gamma', 'method', 'save_image_dir',
            'save_path', 'logits_dir'
        ]
        missing_args = [arg for arg in required_args if arg not in self.args]
        if missing_args:
            logger.error(f"Client {self.client_id}: Missing required arguments: {missing_args}")
            raise ValueError(f"Missing required arguments: {missing_args}")

        # Device configuration
        self.device = torch.device(self.args['device'])

        # Dataset and model configurations (dynamically set)
        self.dataset = self.args['dataset']
        self.model_name = self.args['model']
        self.num_classes = self.args['num_classes']
        self.channel = self.args['channel']
        self.im_size = self.args['im_size']

        # Training configurations
        self.ipc = self.args['ipc']  # Instances Per Class
        self.temperature = self.args['temperature']
        self.gamma = self.args['gamma']
        self.method = self.args['method']

        # Paths for saving data
        self.save_image_dir = self.args['save_image_dir']
        self.save_path = self.args['save_path']
        self.logits_dir = self.args['logits_dir']

        self.save_image_path = os.path.join(self.save_image_dir, f'Client_{self.client_id}')
        self.synthetic_data_path = os.path.join(self.save_path, f'Client_{self.client_id}')
        self.logit_path = os.path.join(self.logits_dir, f'Client_{self.client_id}')

        # Create necessary directories
        os.makedirs(self.save_image_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.synthetic_data_path), exist_ok=True)

        # Initialize synthetic data placeholders
        self.image_syn = None
        self.label_syn = None
        self.global_Vc = []

        # Initialize the global model
        self.model = self.load_global_model()

        # Set normalization parameters based on dataset
        self.mean, self.std = self.set_normalization_parameters()

    def set_normalization_parameters(self) -> tuple:
        """
        Sets the mean and standard deviation based on the dataset.

        Returns:
            tuple: (mean list, std list)
        """
        if self.dataset.upper() == 'CIFAR10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif self.dataset.upper() == 'MNIST':
            mean = [0.1307]
            std = [0.3081]
        else:
            mean = [0.0] * self.channel
            std = [1.0] * self.channel
            logger.warning(f"Client {self.client_id}: Unknown dataset '{self.dataset}'. Using default normalization.")
        return mean, std

    def load_global_model(self) -> torch.nn.Module:
        """
        Loads the latest global model from the server and resamples it.

        Returns:
            torch.nn.Module: Loaded and resampled global model.
        """
        logger.info(f"Client {self.client_id}: Loading global model.")
        try:
            model = load_latest_model(
                model_dir=self.args.get(/home/t914a431/models'),
                model_name=self.model_name,
                channel=self.channel,
                num_classes=self.num_classes,
                im_size=self.im_size,
                device=self.device
            )
            self.resample_model(model)
            model.eval()
            logger.info(f"Client {self.client_id}: Global model loaded and resampled successfully.")
            return model
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error loading global model - {e}")
            raise e

    def resample_model(self, model: torch.nn.Module):
        """
        Resamples the model parameters with a weighted combination of current parameters and random noise.

        Args:
            model (torch.nn.Module): The model to resample.
        """
        logger.info(f"Client {self.client_id}: Resampling model parameters.")
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param)
                param.data = self.gamma * param.data + (1 - self.gamma) * noise
        logger.info(f"Client {self.client_id}: Model parameters resampled.")

    def calculate_and_save_logits(self, round_num: int):
        """
        Calculates class-wise averaged logits and saves them to disk.

        Args:
            round_num (int): Current round number.
        """
        round_logit_path = os.path.join(
            self.logit_path,
            f'Round_{round_num}'
        )
        os.makedirs(round_logit_path, exist_ok=True)

        logger.info(f"Client {self.client_id}: Calculating and saving class-wise logits and soft labels for round {round_num}.")

        try:
            calculate_logits_labels(
                model_net=self.model,
                partition=self.data_partition,
                num_classes=self.num_classes,
                device=self.device,
                path=round_logit_path,
                ipc=self.ipc,
                temperature=self.temperature
            )
            logger.info(f"Client {self.client_id}: Class-wise logits and soft labels calculated and saved.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error calculating and saving logits - {e}")

    def load_global_aggregated_logits(self, round_num: int) -> list:
        """
        Loads the global aggregated logits from the server's global directory for the current round.

        Args:
            round_num (int): Current round number.

        Returns:
            list: Aggregated logits per class.
        """
        global_logits_filename = f'Round{round_num}_Global_Vc.pt'
        global_logits_path = os.path.join(
            self.logits_dir,
            'Global',
            global_logits_filename
        )
        if os.path.exists(global_logits_path):
            try:
                aggregated_tensors = torch.load(global_logits_path, map_location=self.device)
                logger.info(f"Client {self.client_id}: Loaded aggregated logits from {global_logits_path}.")
            except Exception as e:
                logger.error(f"Client {self.client_id}: Error loading aggregated logits - {e}")
                aggregated_tensors = [torch.zeros(self.num_classes, device=self.device) for _ in range(self.num_classes)]
        else:
            logger.warning(f"Client {self.client_id}: Aggregated logits not found at {global_logits_path}. Initializing with zeros.")
            aggregated_tensors = [torch.zeros(self.num_classes, device=self.device) for _ in range(self.num_classes)]
        return aggregated_tensors

    def initialize_synthetic_data(self, round_num: int):
        """
        Initializes the synthetic dataset, optionally using real data for initialization.
    
        Args:
            round_num (int): Current round number.
        """
        logger.info(f"Client {self.client_id}: Initializing synthetic data for round {round_num}.")
        try:
            # Load global aggregated logits for the current round
            self.global_Vc = self.load_global_aggregated_logits(round_num)
    
            # Initialize synthetic images and labels
            self.image_syn = torch.randn(
                size=(self.num_classes * self.ipc, self.channel, *self.im_size),
                dtype=torch.float,
                requires_grad=True,
                device=self.device
            )
            self.label_syn = torch.arange(self.num_classes).repeat(self.ipc).to(self.device, dtype=torch.long)
    
            initialized_classes = []
    
            if self.args.init == 'real':
                logger.info(f"Client {self.client_id}: Initializing synthetic data from real images.")
                for c in range(self.num_classes):
                    real_loader = self.get_images_loader(c, max_batch_size=self.ipc)
                    if real_loader is not None:
                        try:
                            images, _ = next(iter(real_loader))
                            images = images.to(self.device)
                            if images.size(0) >= self.ipc:
                                selected_images = images[:self.ipc]
                                initialized_classes.append(c)
                                self.image_syn.data[c * self.ipc:(c + 1) * self.ipc] = selected_images.detach().data
                                logger.info(f"Client {self.client_id}: Initialized class {c} synthetic images with real data.")
                            else:
                                logger.warning(f"Client {self.client_id}: Not enough images for class {c}. Required at least {self.ipc}, Available: {images.size(0)}. Skipping initialization.")
                        except StopIteration:
                            logger.warning(f"Client {self.client_id}: No images retrieved for class {c} in DataLoader.")
                    else:
                        logger.warning(f"Client {self.client_id}: No real images for class {c}, skipping initialization.")
    
            if not initialized_classes:
                logger.info(f"Client {self.client_id}: No classes initialized with real data. Synthetic data remains randomly initialized.")
    
            self.initialized_classes = initialized_classes
    
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error initializing synthetic data - {e}")
            raise e

    def get_images_loader(self, class_label: int, max_batch_size: int = 256) -> DataLoader or None:
        """
        Retrieves a DataLoader that yields a batch of images from a specified class.

        Args:
            class_label (int): The class label.
            max_batch_size (int): Maximum batch size for loading images.

        Returns:
            DataLoader or None: DataLoader yielding a batch of images, or None if no images are available.
        """
        try:
            # Get all indices for the specified class
            all_labels = np.array(self.data_partition.dataset.targets)[self.data_partition.indices]
            class_indices = np.where(all_labels == class_label)[0]

            if len(class_indices) == 0:
                logger.warning(f"Client {self.client_id}: No images available for class {class_label}.")
                return None

            # Determine actual batch size
            actual_batch_size = min(len(class_indices), max_batch_size)

            # Select indices up to the actual batch size
            selected_indices = class_indices[:actual_batch_size]
            class_subset_indices = [self.data_partition.indices[i] for i in selected_indices]
            class_subset = Subset(self.data_partition.dataset, class_subset_indices)

            # Create a DataLoader for the class subset
            class_loader = DataLoader(class_subset, batch_size=actual_batch_size, shuffle=False)

            return class_loader
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error creating DataLoader for class {class_label} - {e}")
            return None

    def visualize_synthetic_data(self, iteration: int, round_num: int):
        """
        Visualizes and saves the synthetic data.

        Args:
            iteration (int): Current iteration number.
            round_num (int): Current round number.
        """
        try:
            save_name = os.path.join(
                self.save_image_path,
                f'vis_{self.method}_{self.dataset}_{self.model_name}_round{round_num}_client{self.client_id}_{self.ipc}ipc_iter{iteration}.png'
            )
            image_syn_vis = copy.deepcopy(self.image_syn.detach().cpu())

            for ch in range(self.channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * self.std[ch] + self.mean[ch]

            image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
            save_image(image_syn_vis, save_name, nrow=self.ipc)
            logger.info(f"Client {self.client_id}: Synthetic data visualization saved to {save_name}.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error visualizing synthetic data - {e}")

    def save_synthetic_data(self, r):
        logger.info(f"Client {self.client_id}: Saving synthetic data.")
        try:
            # Path to save synthetic data
            savepath = os.path.join(
                self.synthetic_data_path,
                f'res_{self.method}_{self.dataset}_{self.model_name}_Client{self.client_id}_{self.ipc}ipc_Round{r}.pt'
            )
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
    
            # Save only the classes that were properly initialized
            if self.initialized_classes:
                filtered_images = []
                filtered_labels = []
                for c in self.initialized_classes:
                    start_idx = c * self.ipc
                    end_idx = (c + 1) * self.ipc
                    filtered_images.append(self.image_syn[start_idx:end_idx])
                    filtered_labels.append(self.label_syn[start_idx:end_idx])
    
                # Stack filtered images and labels into single tensors
                filtered_images = torch.cat(filtered_images)
                filtered_labels = torch.cat(filtered_labels)
    
                data_save = {
                    'images': filtered_images.detach().cpu(),
                    'labels': filtered_labels.detach().cpu()
                }
    
                torch.save(data_save, savepath)
                logger.info(f"Client {self.client_id}: Synthetic data saved to {savepath}.")
            else:
                logger.warning(f"Client {self.client_id}: No initialized classes with enough real data. Skipping saving synthetic data.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error saving synthetic data - {e}")

    def train_synthetic_data(self, round_num: int):
        """
        Trains the synthetic data using the global aggregated logits.

        Args:
            round_num (int): Current round number.
        """
        logger.info(f"Client {self.client_id}: Starting synthetic data training for round {round_num}.")
        try:
            optimizer_img = optim.SGD([self.image_syn], lr=self.args['lr_img'], momentum=0.5)
            logger.info(f"Client {self.client_id}: Data Condensation begins...")

            for it in range(1, self.args['Iteration'] + 1):
                if it in self.args['eval_it_pool']:
                    self.visualize_synthetic_data(it, round_num)

                # Freeze the model parameters
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

                # Access the 'embed' method
                embed = getattr(self.model, 'embed', None)
                if embed is None:
                    raise AttributeError("Model does not have an 'embed' method.")

                loss = torch.tensor(0.0, device=self.device)

                lambda_cdc = self.dynamic_lambda_cdc(it, self.args['Iteration'] + 1)

                for c in self.initialized_classes:
                    # Retrieve synthetic images for class c
                    img_syn = self.image_syn[c * self.ipc:(c + 1) * self.ipc]
                    if img_syn.size(0) == 0:
                        logger.warning(f"Client {self.client_id}: No synthetic images for class {c}. Skipping.")
                        continue

                    # Compute synthetic logits using the global model
                    logit_syn = self.model(img_syn)
                    local_ukc = torch.mean(logit_syn, dim=0)

                    # Compute synthetic features
                    output_syn = embed(img_syn)
                    mean_feature_syn = torch.mean(output_syn, dim=0)

                    # Retrieve real images DataLoader for class c
                    real_loader = self.get_images_loader(c)
                    if real_loader is not None:
                        try:
                            # Get the first (and only) batch
                            images, _ = next(iter(real_loader))
                            images = images.to(self.device)
                            output_real = embed(images)
                            mean_feature_real = torch.mean(output_real, dim=0)

                            # Distribution Matching Loss
                            loss_feature = torch.sum((mean_feature_real - mean_feature_syn) ** 2)
                            loss += loss_feature
                            logger.debug(f"Client {self.client_id}: Class {c} Distribution Matching Loss: {loss_feature.item():.4f}")

                            # Client Data Condensation Loss
                            if self.global_Vc[c].numel() > 0:
                                loss_logit = lambda_cdc * compute_swd(local_ukc, self.global_Vc[c])
                                loss += loss_logit
                                logger.debug(f"Client {self.client_id}: Class {c} Client Data Condensation Loss: {loss_logit.item():.4f}")
                            else:
                                logger.warning(f"Client {self.client_id}: Missing global logits for class {c}. Skipping SWD computation.")
                        except StopIteration:
                            logger.warning(f"Client {self.client_id}: No images retrieved for class {c} in DataLoader.")
                    else:
                        logger.warning(f"Client {self.client_id}: No real images for class {c}. Skipping feature matching.")

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()

                if it % 10 == 0 or it == self.args['Iteration']:
                    logger.info(f"Client {self.client_id}: Iteration {it}, Loss: {loss.item():.4f}")

            # Save the final synthetic data
            self.save_synthetic_data(round_num)
            logger.info(f"Client {self.client_id}: Synthetic data training completed for round {round_num}.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during synthetic data training - {e}")

    def dynamic_lambda_cdc(self, iteration: int, total_iterations: int) -> float:
        """
        Dynamically adjusts lambda_cdc based on the current iteration.

        Args:
            iteration (int): Current iteration number.
            total_iterations (int): Total number of iterations.

        Returns:
            float: Updated lambda_cdc value.
        """
        max_lambda = 1.0
        min_lambda = 0.1
        lambda_cdc = min_lambda + (max_lambda - min_lambda) * (iteration / total_iterations)
        return lambda_cdc
