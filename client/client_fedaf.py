# client_fedaf.py  (corrected)
#
# Key fixes vs. original:
#   * CDC now actually trains the synthetic data: the SWD term is differentiable
#     (utils_fedaf.compute_swd) and is matched per-class to the FULL aggregated
#     global mean logit v_c (a [C] row of the [C, C] matrix), not a single
#     diagonal scalar reused for every class.
#   * lambda for CDC comes from args.loc_cdc (no hard-coded 0.8).
#   * Model re-sampling is applied to a *fresh* perturbation at EACH condensation
#     step (Eq. 17, "in each step of the local data condensation") on a private
#     copy of the global model; the clean global model is kept for computing the
#     shared V and R statistics.
#   * R soft labels (r_{k,c}) are computed from REAL data with the clean global
#     model (Eq. 11), so the server's LGKM matches real-data knowledge against
#     synthetic-data knowledge (not synthetic-vs-synthetic).
#   * Condensation optimiser uses SGD(momentum=0.9), matching the paper.

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
    ensure_directory_exists,
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Client:
    def __init__(self, client_id, data_partition, args):
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
        self.lambda_cdc = args.loc_cdc          # CDC weight (lambda_loc)
        self.method = args.method
        self.dataset = args.dataset
        self.model_name = args.model_name
        self.save_image_path = os.path.join(args.save_image_dir, f'Client_{self.client_id}')
        self.synthetic_data_path = os.path.join(args.save_path, f'Client_{self.client_id}')

        os.makedirs(self.save_image_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.synthetic_data_path), exist_ok=True)

        self.synthetic_data = []
        self.global_V = None                    # aggregated [C, C] mean-logit matrix
        self.initialized_classes = []
        self.model = None                       # clean global model (for V / R)

    # ----- phase entry points --------------------------------------------- #
    def run_Vkc(self, r):
        """Compute & save v_{k,c} from REAL data using the clean global model."""
        self.calculate_and_save_V_logits(r)

    def run_data_condensation(self, r):
        self.global_V = self.load_global_aggregated_logits(r)
        self.initialize_synthetic_data(r)
        self.train_synthetic_data(r)
        self.save_synthetic_data(r)

    def run_Rkc(self, r):
        """Compute & save r_{k,c} from REAL data (Eq. 11) using the clean model."""
        self.calculate_and_save_R_logits(r)

    # ----- model handling -------------------------------------------------- #
    def load_global_model(self):
        logger.info(f"Client {self.client_id}: Loading global model.")
        model_net = load_latest_model(
            model_dir=self.args.model_dir, model_name=self.model_name,
            channel=self.channel, num_classes=self.num_classes,
            im_size=self.im_size, device=self.device,
        )
        model_net.eval()
        return model_net

    def _perturb_(self, model_net, clean_params):
        """In-place fresh re-sample: w <- gamma*w_clean + (1-gamma)*noise (Eq. 17)."""
        with torch.no_grad():
            for p, cp in zip(model_net.parameters(), clean_params):
                p.data = self.gamma * cp + (1.0 - self.gamma) * torch.randn_like(cp)

    # ----- shared statistics ---------------------------------------------- #
    def calculate_and_save_V_logits(self, r):
        path = os.path.join(self.args.logits_dir, f'Client_{self.client_id}', f'Round_{r}')
        os.makedirs(path, exist_ok=True)
        calculate_logits_labels(
            model_net=self.model, partition=self.data_partition,
            num_classes=self.num_classes, device=self.device, path=path,
            ipc=self.ipc, temperature=self.temperature, logits_type='V',
        )
        logger.info(f"Client {self.client_id}: Vkc (real-data mean logits) saved.")

    def calculate_and_save_R_logits(self, r):
        path = os.path.join(self.args.logits_dir, f'Client_{self.client_id}', f'Round_{r}')
        os.makedirs(path, exist_ok=True)
        # IMPORTANT: real data + clean global model (NOT synthetic data).
        calculate_logits_labels(
            model_net=self.model, partition=self.data_partition,
            num_classes=self.num_classes, device=self.device, path=path,
            ipc=self.ipc, temperature=self.temperature, logits_type='R',
        )
        logger.info(f"Client {self.client_id}: Rkc (real-data soft labels) saved.")

    def load_global_aggregated_logits(self, r):
        """Load the aggregated GLOBAL mean-logit matrix V = [v_0..v_{C-1}], shape [C, C]."""
        path = os.path.join(self.args.logits_dir, 'Global', f'Round{r}_Global_Vc.pt')
        if os.path.exists(path):
            V = torch.load(path, map_location=self.device)
            if isinstance(V, torch.Tensor) and V.shape == (self.num_classes, self.num_classes):
                logger.info(f"Client {self.client_id}: Loaded global V matrix {tuple(V.shape)}.")
                return V
            logger.error(f"Client {self.client_id}: global V has wrong shape "
                         f"{getattr(V, 'shape', None)}; expected [{self.num_classes},{self.num_classes}].")
        else:
            logger.warning(f"Client {self.client_id}: global V not found at {path}.")
        return torch.zeros(self.num_classes, self.num_classes, device=self.device)

    # ----- synthetic data -------------------------------------------------- #
    def initialize_synthetic_data(self, r):
        logger.info(f"Client {self.client_id}: Initializing synthetic data.")
        self.image_syn = torch.randn(
            size=(self.num_classes * self.ipc, self.channel, self.im_size[0], self.im_size[1]),
            dtype=torch.float, requires_grad=True, device=self.device,
        )
        self.label_syn = torch.arange(self.num_classes, dtype=torch.long,
                                      device=self.device).repeat_interleave(self.ipc)
        initialized_classes = []
        if self.args.init == 'real':
            for c in range(self.num_classes):
                loader = self.get_images_loader(c, max_batch_size=self.ipc)
                if loader is not None:
                    images, _ = next(iter(loader))
                    images = images.to(self.device)
                    if images.size(0) >= self.ipc:
                        self.image_syn.data[c * self.ipc:(c + 1) * self.ipc] = images[:self.ipc].detach().data
                        initialized_classes.append(c)
        if not initialized_classes:
            logger.info(f"Client {self.client_id}: No class initialised from real data.")
        self.initialized_classes = initialized_classes

    def get_images_loader(self, class_label, max_batch_size=256):
        try:
            all_labels = np.array(self.data_partition.dataset.targets)[self.data_partition.indices]
            class_indices = np.where(all_labels == class_label)[0]
            if len(class_indices) == 0:
                return None
            actual = min(len(class_indices), max_batch_size)
            sel = np.random.choice(class_indices, size=actual, replace=False)
            subset_idx = [self.data_partition.indices[i] for i in sel]
            return DataLoader(Subset(self.data_partition.dataset, subset_idx),
                              batch_size=actual, shuffle=False)
        except Exception as e:
            logger.error(f"Client {self.client_id}: loader error for class {class_label} - {e}")
            return None

    def visualize_synthetic_data(self, iteration, mean, std, r):
        try:
            save_name = os.path.join(
                self.save_image_path,
                f'vis_{self.method}_{self.dataset}_{self.model_name}_round{r}_client{self.client_id}_{self.ipc}ipc_iter{iteration}.png')
            vis = copy.deepcopy(self.image_syn.detach().cpu())
            for ch in range(self.channel):
                vis[:, ch] = vis[:, ch] * std[ch] + mean[ch]
            save_image(torch.clamp(vis, 0.0, 1.0), save_name, nrow=self.ipc)
        except Exception as e:
            logger.error(f"Client {self.client_id}: visualisation error - {e}")

    def synthetic_save_path(self, r):
        """Canonical path of this client's condensed dataset for round r.

        Used both when saving and by the restart logic to detect a finished
        condensation phase, so the two never drift apart.
        """
        return os.path.join(
            self.synthetic_data_path,
            f'res_{self.method}_{self.dataset}_{self.model_name}_Client{self.client_id}_{self.ipc}ipc_Round{r}.pt')

    def save_synthetic_data(self, r):
        savepath = self.synthetic_save_path(r)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        if not self.initialized_classes:
            logger.warning(f"Client {self.client_id}: nothing to save.")
            return
        imgs, lbls = [], []
        for c in self.initialized_classes:
            imgs.append(self.image_syn[c * self.ipc:(c + 1) * self.ipc])
            lbls.append(self.label_syn[c * self.ipc:(c + 1) * self.ipc])
        torch.save({'images': torch.cat(imgs).detach().cpu(),
                    'labels': torch.cat(lbls).detach().cpu()}, savepath)
        logger.info(f"Client {self.client_id}: synthetic data saved to {savepath}.")

    # ----- collaborative data condensation --------------------------------- #
    def train_synthetic_data(self, r):
        """Minimise  L_DM + lambda_loc * sum_c SWD(u_{k,c}(S), v_c)   (Eq. 4 + Eq. 10)."""
        logger.info(f"Client {self.client_id}: starting data condensation.")
        optimizer_img = optim.SGD([self.image_syn], lr=self.args.lr_img, momentum=0.9)

        # Private copy that gets re-perturbed every step; self.model stays clean.
        work_model = copy.deepcopy(self.model).to(self.device)
        work_model.eval()
        for p in work_model.parameters():
            p.requires_grad_(False)
        clean_params = [p.detach().clone() for p in work_model.parameters()]
        embed = getattr(work_model, 'embed', None)
        if embed is None:
            raise AttributeError("Model has no 'embed' method.")

        have_global_V = self.global_V is not None and self.global_V.abs().sum() > 0

        for it in range(1, self.args.Iteration + 1):
            if it in self.args.eval_it_pool:
                self.visualize_synthetic_data(it, self.args.mean, self.args.std, r)

            # fresh re-sample THIS step
            self._perturb_(work_model, clean_params)

            loss = torch.tensor(0.0, device=self.device)
            for c in self.initialized_classes:
                img_syn = self.image_syn[c * self.ipc:(c + 1) * self.ipc]
                if img_syn.size(0) == 0:
                    continue

                # --- distribution matching loss (Eq. 4) ---
                feat_syn = embed(img_syn).mean(dim=0)
                real_loader = self.get_images_loader(c, max_batch_size=self.ipc)
                if real_loader is None:
                    continue
                images, _ = next(iter(real_loader))
                feat_real = embed(images.to(self.device)).mean(dim=0)
                loss_feature = torch.sum((feat_real - feat_syn) ** 2)

                # --- collaborative data condensation loss (Eq. 10, SWD) ---
                if have_global_V:
                    logit_syn = work_model(img_syn)                 # [M, C], differentiable
                    v_c = self.global_V[c].detach()                 # [C] target (point mass)
                    v_c_rep = v_c.unsqueeze(0).expand(logit_syn.size(0), -1)
                    loss_logit = compute_swd(logit_syn, v_c_rep)
                    loss = loss + loss_feature + self.lambda_cdc * loss_logit
                else:
                    loss = loss + loss_feature

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            if it % 100 == 0 or it == self.args.Iteration:
                logger.info(f"Client {self.client_id}: iter {it}, loss {loss.item():.4f}")

        logger.info(f"Client {self.client_id}: data condensation completed.")