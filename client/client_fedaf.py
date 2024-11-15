# client_fedaf.py

import os
import copy
import numpy as np
import torch
from torchvision.utils import save_image
from torch import optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from utils.utils_fedaf import (
    load_latest_model,
    calculate_logits_labels,
    compute_swd,
    get_base_dataset,
    ensure_directory_exists
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Client:
    def __init__(self, client_id, data_partition, args):
        """
        Initializes a federated learning client.

        Args:
            client_id (int): Unique identifier for the client.
            data_partition (torch.utils.data.Subset): Subset of the dataset assigned to the client.
            args (Namespace): Argument parser containing hyperparameters and configurations.
        """
        self.client_id = client_id
        self.data_partition = data_partition
        self.args = args
        self.device = torch.device(args.device)
        self.num_classes = args.num_classes
        self.channel = args.channel
        self.im_size = args.im_size
        self.ipc = args.ipc
        self.temperature = args.temperature
        self.gamma = args.gamma
        self.method = args.method
        self.dataset = args.dataset
        self.model_name = args.model_name
        self.save_image_path = os.path.join(args.save_image_dir, f'Client_{self.client_id}')

        self.synthetic_data_path = os.path.join(args.save_path, f'Client_{self.client_id}')

        os.makedirs(self.save_image_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.synthetic_data_path), exist_ok=True)

        self.synthetic_data = []
        self.global_Vc = []
        self.initialized_classes = []
        self.model = None  # Will be loaded in run methods

    def run_Vkc(self, r):
        """
        Runs the computation of Vkc logits.
        """
        # The model is already loaded and resampled in client_full_round
        # Calculate and save Vkc
        self.calculate_and_save_V_logits(r)

    def run_data_condensation(self, r):
        """
        Runs data condensation.
        """
        # Load global aggregated logits
        self.global_Vc = self.load_global_aggregated_logits(r)

        # Initialize synthetic data
        self.initialize_synthetic_data(r)

        # Train synthetic data
        self.train_synthetic_data(r)

        # Save synthetic data
        self.save_synthetic_data(r)
        
    def dynamic_lambda_cdc(self, current_iter, total_iters):
        """
        Dynamically adjusts the lambda_cdc value based on the current iteration.
    
        Args:
            current_iter (int): The current iteration number.
            total_iters (int): The total number of iterations.
    
        Returns:
            float: The dynamically adjusted lambda_cdc value.
        """
        # Example: Linear schedule decreasing lambda_cdc over time
        # initial_lambda_cdc = self.args.loc_cdc  # Initial value from args
        # lambda_cdc = initial_lambda_cdc * (1 - current_iter / total_iters)
        lambda_cdc = 0.8
        return lambda_cdc
 
    def run_Rkc(self, r):
        """
        Runs the computation of Rkc logits.
        """
        # Load synthetic data
        self.load_synthetic_data(r)

        # Calculate and save Rkc
        self.calculate_and_save_R_logits(r)

    def load_global_model(self):
        """
        Loads the latest global model from the server.

        Returns:
            torch.nn.Module: Loaded global model.
        """
        logger.info(f"Client {self.client_id}: Loading global model.")
        try:
            model_net = load_latest_model(
                model_dir=self.args.model_dir,
                model_name=self.model_name,
                channel=self.channel,
                num_classes=self.num_classes,
                im_size=self.im_size,
                device=self.device
            )
            model_net.eval()
            logger.info(f"Client {self.client_id}: Global model loaded successfully.")
            return model_net
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error loading global model - {e}")
            raise e

    def resample_model(self, model_net):
        """
        Resamples the model parameters with a weighted combination of current parameters and random noise.

        Args:
            model_net (torch.nn.Module): The model to resample.
        """
        logger.info(f"Client {self.client_id}: Resampling model parameters.")
        with torch.no_grad():
            for param in model_net.parameters():
                noise = torch.randn_like(param)
                param.data = self.gamma * param.data + (1 - self.gamma) * noise
        logger.info(f"Client {self.client_id}: Model parameters resampled.")
        self.model = model_net  # Update the model

    def calculate_and_save_V_logits(self, r):
        """
        Calculates and saves class-wise averaged logits Vkc before data condensation.
        """
        self.logit_path = os.path.join(
            self.args.logits_dir, f'Client_{self.client_id}', f'Round_{r}'
        )
        os.makedirs(self.logit_path, exist_ok=True)

        logger.info(f"Client {self.client_id}: Calculating and saving Vkc logits.")
        try:
            calculate_logits_labels(
                model_net=self.model,
                partition=self.data_partition,
                num_classes=self.num_classes,
                device=self.device,
                path=self.logit_path,
                ipc=self.ipc,
                temperature=self.temperature,
                logits_type='V'  # Specify that we are calculating V logits
            )
            logger.info(f"Client {self.client_id}: Vkc logits calculated and saved.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error calculating and saving Vkc logits - {e}")

    def calculate_and_save_R_logits(self, r):
        """
        Calculates and saves class-wise averaged logits Rkc after data condensation.
        """
        self.logit_path = os.path.join(
            self.args.logits_dir, f'Client_{self.client_id}', f'Round_{r}'
        )
        os.makedirs(self.logit_path, exist_ok=True)

        logger.info(f"Client {self.client_id}: Calculating and saving Rkc logits using synthetic data.")
        try:
            calculate_logits_labels(
                model_net=self.model,
                partition=self.synthetic_data,  # Use synthetic data
                num_classes=self.num_classes,
                device=self.device,
                path=self.logit_path,
                ipc=self.ipc,
                temperature=self.temperature,
                logits_type='R'  # Specify that we are calculating R logits
            )
            logger.info(f"Client {self.client_id}: Rkc logits calculated and saved.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error calculating and saving Rkc logits - {e}")

    def load_synthetic_data(self, r):
        """
        Loads the synthetic data saved after data condensation.
        """
        savepath = os.path.join(
            self.synthetic_data_path,
            f'res_{self.method}_{self.dataset}_{self.model_name}_Client{self.client_id}_{self.ipc}ipc_Round{r}.pt'
        )
        try:
            data_save = torch.load(savepath, map_location=self.device)
            self.synthetic_data = DataLoader(
                TensorDataset(data_save['images'].to(self.device), data_save['labels'].to(self.device)),
                batch_size=self.ipc,
                shuffle=False
            )
            logger.info(f"Client {self.client_id}: Synthetic data loaded from {savepath}.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error loading synthetic data - {e}")
            raise e

    def load_global_aggregated_logits(self, r):
        """
        Loads the global aggregated logits from the server's global directory for the current round.

        Args:
            r (int): Current round number.

        Returns:
            list of torch.Tensor: Aggregated logits Rc_list, where each Rc_list[c] is a tensor of shape [num_classes,].
        """
        global_logits_filename = f'Round{r}_Global_Vc.pt'
        global_logits_path = os.path.join(self.args.logits_dir, 'Global', global_logits_filename)
        if os.path.exists(global_logits_path):
            try:
                aggregated_logits = torch.load(global_logits_path, map_location=self.device)  # Tensor of shape [num_classes,]
                if isinstance(aggregated_logits, torch.Tensor) and aggregated_logits.shape[0] == self.num_classes:
                    logger.info(f"Client {self.client_id}: Loaded aggregated logits from {global_logits_path}.")
                else:
                    logger.error(f"Client {self.client_id}: Aggregated logits format incorrect. Expected tensor of shape [{self.num_classes}], got {aggregated_logits.shape}.")
                    aggregated_logits = torch.zeros(self.num_classes, device=self.device)
            except Exception as e:
                logger.error(f"Client {self.client_id}: Error loading aggregated logits - {e}")
                aggregated_logits = torch.zeros(self.num_classes, device=self.device)
        else:
            logger.warning(f"Client {self.client_id}: Aggregated logits not found at {global_logits_path}. Initializing with zeros.")
            aggregated_logits = torch.zeros(self.num_classes, device=self.device)
        return aggregated_logits

    def initialize_synthetic_data(self, r):
        """
        Initializes the synthetic dataset, optionally using real data for initialization.
        """
        logger.info(f"Client {self.client_id}: Initializing synthetic data.")
        try:
            # Initialize synthetic images and labels
            self.image_syn = torch.randn(
                size=(self.num_classes * self.ipc, self.channel, self.im_size[0], self.im_size[1]),
                dtype=torch.float,
                requires_grad=True,
                device=self.device
            )
            self.label_syn = torch.arange(self.num_classes, dtype=torch.long, device=self.device).repeat_interleave(self.ipc)

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

    def get_images_loader(self, class_label, max_batch_size=256):
        """
        Retrieves a DataLoader that yields a batch of images from a specified class.
        If the number of available images is less than max_batch_size, use all available images.

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
            selected_indices = np.random.choice(class_indices, size=actual_batch_size, replace=False)

            # Create a Subset for the selected indices
            class_subset_indices = [self.data_partition.indices[i] for i in selected_indices]
            class_subset = Subset(self.data_partition.dataset, class_subset_indices)

            # Create a DataLoader for the class subset
            class_loader = DataLoader(class_subset, batch_size=actual_batch_size, shuffle=False)

            return class_loader
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error creating DataLoader for class {class_label} - {e}")
            return None

    def visualize_synthetic_data(self, iteration, mean, std, r):
        """
        Visualizes and saves the synthetic data.

        Args:
            iteration (int): Current iteration number.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
            r (int): Current round number.
        """
        try:
            save_name = os.path.join(
                self.save_image_path,
                f'vis_{self.args.method}_{self.args.dataset}_{self.model_name}_round{r}_client{self.client_id}_{self.ipc}ipc_iter{iteration}.png'
            )
            image_syn_vis = copy.deepcopy(self.image_syn.detach().cpu())

            for ch in range(self.channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]

            image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
            save_image(image_syn_vis, save_name, nrow=self.ipc)
            logger.info(f"Client {self.client_id}: Synthetic data visualization saved to {save_name}.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error visualizing synthetic data - {e}")

    def save_synthetic_data(self, r):
        """
        Saves the synthetic dataset to disk, excludes non-initialized classes.
        """
        logger.info(f"Client {self.client_id}: Saving synthetic data.")
        try:
            # Path to save synthetic data
            savepath = os.path.join(
                self.synthetic_data_path,
                f'res_{self.method}_{self.dataset}_{self.model_name}_Client{self.client_id}_{self.ipc}ipc_Round{r}.pt'
            )
            os.makedirs(os.path.dirname(savepath), exist_ok=True)

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
                logger.warning(f"Client {self.client_id}: No initialized classes to save. Skipping saving synthetic data.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error saving synthetic data - {e}")

    def train_synthetic_data(self, r):
        """
        Trains the synthetic data using the global aggregated logits.
        """
        logger.info(f"Client {self.client_id}: Starting synthetic data training.")
        try:
            optimizer_img = optim.SGD([self.image_syn], lr=self.args.lr_img)
            logger.info("[X] Data Condensation begins...")

            for it in range(1, self.args.Iteration + 1):
                if it in self.args.eval_it_pool:
                    # Visualization and saving synthetic images
                    self.visualize_synthetic_data(it, self.args.mean, self.args.std, r)

                # Freeze the model parameters
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

                # Access the 'embed' method
                embed = getattr(self.model, 'embed', None)
                if embed is None:
                    raise AttributeError("Model does not have an 'embed' method.")

                loss = torch.tensor(0.0, device=self.device)

                lambda_cdc = self.dynamic_lambda_cdc(it, self.args.Iteration + 1)

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
                    real_loader = self.get_images_loader(c, max_batch_size=self.ipc)
                    if real_loader is not None:
                        try:
                            # Get the first (and only) batch
                            images, _ = next(iter(real_loader))
                            images = images.to(self.device)
                            output_real = embed(images)
                            mean_feature_real = torch.mean(output_real, dim=0)

                            # Distribution Matching Loss
                            loss_feature = torch.sum((mean_feature_real - mean_feature_syn) ** 2)
                            logger.debug(f"Client {self.client_id}: Class {c} Distribution Matching Loss: {loss_feature.item():.4f}")

                            # Client Data Condensation Loss
                            if self.global_Vc is not None and self.global_Vc.numel() > 0:
                                loss_logit = compute_swd(local_ukc, self.global_Vc)
                                loss += loss_feature + lambda_cdc * loss_logit
                                logger.debug(f"Client {self.client_id}: Class {c} Client Data Condensation Loss: {loss_logit.item():.4f}")
                            else:
                                logger.warning(f"Client {self.client_id}: Missing global logits. Skipping SWD computation.")
                        except StopIteration:
                            logger.warning(f"Client {self.client_id}: No images retrieved for class {c} in DataLoader.")
                    else:
                        logger.warning(f"Client {self.client_id}: No real images for class {c}. Skipping feature matching.")

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()

                if it % 10 == 0 or it == self.args.Iteration:
                    logger.info(f"Client {self.client_id}: Iteration {it}, Loss: {loss.item():.4f}")

            logger.info(f"Client {self.client_id}: Synthetic data training completed.")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during synthetic data training - {e}")
