# utils_fedaf.py  (corrected)
#
# Key fixes vs. original:
#   * compute_swd is now a *real*, fully-differentiable Sliced Wasserstein
#     Distance implemented in torch (no .detach()/numpy -> gradients now flow
#     back to the synthetic images, so the CDC term actually trains the data).
#   * calculate_logits_labels:
#       - 'R' soft labels are softmax-of-the-mean-logit (Eq. 11), not
#         mean-of-softmax, and are computed from REAL data (see client).
#       - per-class results are saved as a dict {'value': [C], 'count': n} so
#         the server can build the FULL [C, C] class-logit matrix and average
#         only over clients that actually own each class.
#       - a class is averaged over *all* its available samples (no IPC
#         threshold that silently zeroed sparse classes under heavy skew).

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
from config import DATASET_REGISTRY, TORCHVISION_DATASET
from utils.networks import (
    MLP, ConvNet, LeNet, AlexNet, AlexNetBN,
    VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Sliced Wasserstein Distance (differentiable, torch-native)                  #
# --------------------------------------------------------------------------- #
def _match_quantiles(sorted_vals, n):
    """Linearly resample sorted samples (along dim 0) to `n` quantiles.
    sorted_vals: [N, P] already sorted along dim 0. Returns [n, P]."""
    N = sorted_vals.shape[0]
    if N == n:
        return sorted_vals
    x = sorted_vals.transpose(0, 1).unsqueeze(1)          # [P, 1, N]
    x = F.interpolate(x, size=n, mode='linear', align_corners=True)
    return x.squeeze(1).transpose(0, 1)                    # [n, P]


def compute_swd(x, y, num_projections=128):
    """Squared Sliced-Wasserstein-2 distance between two empirical samples.

    Args:
        x, y: torch tensors of shape [N, D] (a set of N points in R^D) or [D]
              (treated as a single point, i.e. a point mass).
        num_projections: number of random 1-D slices.

    Returns:
        Scalar torch tensor (keeps the autograd graph of x / y).

    Note on FedAF usage: peers only share the class mean v_c, so on the global
    side we pass a point mass (v_c tiled to the synthetic batch size). SWD to a
    point mass equals the mean squared logit deviation toward v_c -- exactly the
    'pull synthetic class-c logits toward the aggregated global logit' that the
    CDC term is meant to do, while preserving per-class correspondence.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    D = x.shape[1]
    proj = torch.randn(D, num_projections, device=x.device, dtype=x.dtype)
    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-12)   # unit directions

    x_proj = x @ proj                                        # [Nx, P]
    y_proj = y @ proj                                        # [Ny, P]

    x_sorted, _ = torch.sort(x_proj, dim=0)
    y_sorted, _ = torch.sort(y_proj, dim=0)

    if x_sorted.shape[0] != y_sorted.shape[0]:
        n = max(x_sorted.shape[0], y_sorted.shape[0])
        x_sorted = _match_quantiles(x_sorted, n)
        y_sorted = _match_quantiles(y_sorted, n)

    return ((x_sorted - y_sorted) ** 2).mean()


# --------------------------------------------------------------------------- #
#  Class-wise logits / soft labels                                             #
# --------------------------------------------------------------------------- #
def calculate_logits_labels(model_net, partition, num_classes, device, path,
                            ipc, temperature, logits_type='V'):
    """Compute and save class-wise statistics.

    'V': per-class MEAN LOGIT vector  v_{k,c}            (Eq. 5)   -> [C]
    'R': per-class SOFT LABEL          r_{k,c}=softmax(v_{k,c}/T)  (Eq. 11) -> [C]

    Saved per class c as a dict {'value': tensor[C], 'count': int} in
    f'{logits_type}kc_{c}.pt'. `count` is the number of real samples of class c
    owned by this client (0 => class absent, skipped during aggregation).
    """
    os.makedirs(path, exist_ok=True)

    dataloader = partition if isinstance(partition, DataLoader) \
        else DataLoader(partition, batch_size=256, shuffle=False)

    sum_logits = [torch.zeros(num_classes, device=device) for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]

    model_net.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model_net(images)                       # [B, C] (pre-softmax)
            for i in range(labels.size(0)):
                c = labels[i].item()
                sum_logits[c] += logits[i]
                counts[c] += 1

    for c in range(num_classes):
        if counts[c] > 0:
            mean_logit = sum_logits[c] / counts[c]           # v_{k,c}
            if logits_type == 'V':
                value = mean_logit
            elif logits_type == 'R':
                value = F.softmax(mean_logit / temperature, dim=0)   # softmax-of-mean
            else:
                raise ValueError("logits_type must be 'V' or 'R'")
        else:
            value = torch.zeros(num_classes, device=device)
        try:
            torch.save({'value': value.detach().cpu(), 'count': int(counts[c])},
                       os.path.join(path, f'{logits_type}kc_{c}.pt'))
        except Exception as e:
            logger.error(f"Error saving {logits_type}kc logits for class {c}: {e}")


# --------------------------------------------------------------------------- #
#  Model / dataset helpers (unchanged behaviour)                               #
# --------------------------------------------------------------------------- #
def load_latest_model(model_dir, model_name, channel, num_classes, im_size, device):
    try:
        if os.path.exists(model_dir) and os.listdir(model_dir):
            model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
                           if f.endswith('.pth') and f.startswith('fedaf')]
            latest_model_file = max(model_files, key=os.path.getmtime, default=None)
            if latest_model_file:
                net = get_network(model_name, channel, num_classes, im_size).to(device)
                net.load_state_dict(torch.load(latest_model_file, map_location=device))
                logger.info(f"Loaded model from {latest_model_file}")
                return net
        logger.info("Model directory empty/no valid model. Initializing a new model.")
        return get_network(model_name, channel, num_classes, im_size).to(device)
    except Exception as e:
        logger.error(f"Error loading the latest model: {e}")
        return get_network(model_name, channel, num_classes, im_size).to(device)


def build_transform(dataset_name):
    """Normalisation transform driven by the single dataset registry."""
    meta = DATASET_REGISTRY[dataset_name]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=meta['mean'], std=meta['std']),
    ])


def get_base_dataset(cfg, train=True):
    """Train/test dataset for cfg.dataset, normalised per the registry.

    Supports every dataset in config.DATASET_REGISTRY (MNIST, FMNIST, CIFAR10,
    CIFAR100). The torchvision class is resolved through TORCHVISION_DATASET so
    new datasets only need a registry entry, not edits here.
    """
    if cfg.dataset not in TORCHVISION_DATASET:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    ds_cls = getattr(datasets, TORCHVISION_DATASET[cfg.dataset])
    return ds_cls(cfg.data_path, train=train, download=True,
                  transform=build_transform(cfg.dataset))


def get_test_loader(cfg, batch_size=256, num_workers=0):
    """Normalised test-set DataLoader for cfg.dataset (shared by server/plot)."""
    test_set = get_base_dataset(cfg, train=False)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_network(model, channel, num_classes, im_size=(32, 32)):
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width,
                      net_depth=net_depth, net_act=net_act, net_norm=net_norm,
                      net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)
    else:
        exit('unknown model: %s' % model)
    return net.to('cpu')


def get_default_convnet_setting():
    # NOTE: the paper states BatchNorm ("to facilitate comparison with FedBN").
    # The DM/DC default (and the more stable choice for condensation) is
    # InstanceNorm. To reproduce the paper's stated setup exactly, change
    # net_norm to 'batchnorm' HERE *and* in utils/utils_fedavg.py so FedAF and
    # FedAvg share the same architecture.
    net_width = 128
    net_depth = 3
    net_act = 'relu'
    net_norm = 'instancenorm'
    net_pooling = 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_aggregated_logits(aggregated_logits, args, r, v_r):
    """Save the aggregated [C, C] matrix (V or R) to the shared Global dir."""
    logits_dir = os.path.join(args.logits_dir, 'Global')
    os.makedirs(logits_dir, exist_ok=True)
    path = os.path.join(logits_dir, f'Round{r}_Global_{v_r}c.pt')
    torch.save(aggregated_logits, path)
    logger.info(f"Aggregated {v_r} logits saved to {path} with shape {tuple(aggregated_logits.shape)}.")