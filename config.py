# config.py
#
# Single source of truth for FedAF / FedAvg.
#
#   * config.yaml holds every user-editable setting (see that file).
#   * DATASET_REGISTRY holds the per-dataset constants (channels, classes,
#     image size, normalisation stats, default synthetic-image LR).
#   * load_config() merges the two (+ optional CLI / env overrides) into one
#     fully-resolved, picklable `Config` object that every script consumes, so
#     a value set in one place propagates to fedaf, fedavg and plotting alike.
#
# The Config object is intentionally a plain dataclass of primitive types so it
# pickles cleanly across ProcessPoolExecutor workers (use to_dict / from_dict
# at the process boundary; device is stored as a string for the same reason).

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass, field, asdict, fields
from typing import List, Tuple

import yaml


# --------------------------------------------------------------------------- #
#  Per-dataset constants (the paper's label-skew datasets + MNIST)             #
#    lr_img: default synthetic-image learning rate from the FedAF paper.       #
# --------------------------------------------------------------------------- #
DATASET_REGISTRY = {
    'MNIST':    {'channel': 1, 'num_classes': 10,  'im_size': (28, 28),
                 'mean': [0.1307], 'std': [0.3081], 'lr_img': 0.2},
    'FMNIST':   {'channel': 1, 'num_classes': 10,  'im_size': (28, 28),
                 'mean': [0.2860], 'std': [0.3530], 'lr_img': 0.2},
    'CIFAR10':  {'channel': 3, 'num_classes': 10,  'im_size': (32, 32),
                 'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010],
                 'lr_img': 1.0},
    'CIFAR100': {'channel': 3, 'num_classes': 100, 'im_size': (32, 32),
                 'mean': [0.5071, 0.4865, 0.4409], 'std': [0.2673, 0.2564, 0.2762],
                 'lr_img': 0.1},
}

# Map each registry name to its torchvision dataset class name. Kept here so the
# dataset list lives in exactly one place; utils modules look it up.
TORCHVISION_DATASET = {
    'MNIST': 'MNIST',
    'FMNIST': 'FashionMNIST',
    'CIFAR10': 'CIFAR10',
    'CIFAR100': 'CIFAR100',
}


@dataclass
class Config:
    # ---- what to run ----
    dataset: str = 'CIFAR10'
    model: str = 'ConvNet'
    model_name: str = 'ConvNet'          # alias kept for client/server code
    method: str = 'fedaf'                # set per-script: 'fedaf' or 'fedavg'
    device: str = 'cpu'
    seed: int = 42

    # ---- federated setup ----
    num_clients: int = 10
    alpha: float = 0.1
    rounds: int = 20
    honesty_ratio: float = 1
    resume: bool = True

    # ---- dataset-derived (filled from DATASET_REGISTRY) ----
    channel: int = 3
    num_classes: int = 10
    im_size: Tuple[int, int] = (32, 32)
    mean: List[float] = field(default_factory=lambda: [0.0])
    std: List[float] = field(default_factory=lambda: [1.0])

    # ---- FedAF ----
    ipc: int = 50
    Iteration: int = 1000                # local condensation steps (capital I: client reads args.Iteration)
    eval_mode: str = 'SS'
    eval_it_pool: List[int] = field(default_factory=list)
    global_epochs: int = 500
    global_lr: float = 0.001
    server_batch_size: int = 256
    loc_cdc: float = 0.0001
    loc_lgkm: float = 0.01
    temperature: float = 2.0
    gamma: float = 0.9
    init: str = 'real'
    lr_img: float = 1.0

    # ---- FedAvg ----
    local_epochs: int = 10
    lr: float = 0.01
    batch_size: int = 64

    # ---- paths (absolute, resolved against repo root) ----
    data_path: str = './data'
    logits_dir: str = './logits'
    save_image_dir: str = './images'
    save_path: str = './result'
    models_root: str = './models'
    plots_dir: str = './plots'
    model_dir: str = ''                  # models_root/dataset/model/num_clients/honesty

    # ---- plotting ----
    plot_methods: List[str] = field(default_factory=lambda: ['fedaf', 'fedavg'])
    plot_test_repeats: int = 5

    # ----- (de)serialisation across process boundaries ----- #
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        known = {f.name for f in fields(cls)}
        obj = cls(**{k: v for k, v in d.items() if k in known})
        obj.im_size = tuple(obj.im_size)
        return obj

    # ----- convenience ----- #
    def method_model_path(self, method: str, round_num: int) -> str:
        """Path of the saved global model for `method` at `round_num`."""
        return os.path.join(self.model_dir, f'{method}_global_model_{round_num}.pth')

    def all_dirs(self) -> List[str]:
        return [self.data_path, self.logits_dir, self.save_image_dir,
                self.save_path, self.model_dir, self.plots_dir]

    def ensure_dirs(self) -> None:
        for d in self.all_dirs():
            os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Loading / merging                                                           #
# --------------------------------------------------------------------------- #
def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(project_root(), p))


def _honesty_str(hr) -> str:
    """Format honesty_ratio for the model directory (1.0 -> '1', 0.5 -> '0.5')."""
    f = float(hr)
    return str(int(f)) if f.is_integer() else str(f)


def _resolve_device(dev: str) -> str:
    if dev and dev != 'auto':
        return dev
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def _cli_overrides(argv) -> dict:
    """Parse the small set of fields worth overriding on the command line.

    Only keys the user actually passed are returned, so the YAML stays the
    single source of truth and the CLI is a thin override layer.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--config')
    p.add_argument('--dataset')
    p.add_argument('--model')
    p.add_argument('--device')
    p.add_argument('--seed', type=int)
    p.add_argument('--num-clients', dest='num_clients', type=int)
    p.add_argument('--alpha', type=float)
    p.add_argument('--rounds', type=int)
    p.add_argument('--ipc', type=int)
    resume = p.add_mutually_exclusive_group()
    resume.add_argument('--resume', dest='resume', action='store_true', default=None)
    resume.add_argument('--no-resume', dest='resume', action='store_false', default=None)
    known, _ = p.parse_known_args(argv)
    return {k: v for k, v in vars(known).items() if v is not None and k != 'config'}, getattr(known, 'config', None)


def load_config(method: str = 'fedaf', argv=None) -> Config:
    """Build the fully-resolved Config from config.yaml (+ CLI / env overrides)."""
    overrides, cli_config = _cli_overrides(argv)

    cfg_path = cli_config or os.environ.get('FEDAF_CONFIG') \
        or os.path.join(project_root(), 'config.yaml')
    with open(cfg_path, 'r') as fh:
        raw = yaml.safe_load(fh) or {}

    fedaf = raw.get('fedaf', {}) or {}
    fedavg = raw.get('fedavg', {}) or {}
    paths = raw.get('paths', {}) or {}
    plot = raw.get('plot', {}) or {}

    # ---- top-level + per-method values (CLI overrides win) ----
    dataset = overrides.get('dataset', raw.get('dataset', 'CIFAR10'))
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Known: {sorted(DATASET_REGISTRY)}")
    meta = DATASET_REGISTRY[dataset]
    model = overrides.get('model', raw.get('model', 'ConvNet'))

    lr_img = fedaf.get('lr_img', None)
    lr_img = meta['lr_img'] if lr_img is None else float(lr_img)

    honesty_ratio = raw.get('honesty_ratio', 1)
    num_clients = overrides.get('num_clients', raw.get('num_clients', 10))
    models_root = _resolve_path(paths.get('models', './models'))
    model_dir = os.path.join(models_root, dataset, model,
                             str(num_clients), _honesty_str(honesty_ratio))

    cfg = Config(
        dataset=dataset,
        model=model,
        model_name=model,
        method=method,
        device=_resolve_device(overrides.get('device', raw.get('device', 'auto'))),
        seed=overrides.get('seed', raw.get('seed', 42)),

        num_clients=num_clients,
        alpha=overrides.get('alpha', raw.get('alpha', 0.1)),
        rounds=overrides.get('rounds', raw.get('rounds', 20)),
        honesty_ratio=honesty_ratio,
        resume=overrides.get('resume', raw.get('resume', True)),

        channel=meta['channel'],
        num_classes=meta['num_classes'],
        im_size=tuple(meta['im_size']),
        mean=list(meta['mean']),
        std=list(meta['std']),

        ipc=overrides.get('ipc', fedaf.get('ipc', 50)),
        Iteration=fedaf.get('iteration', 1000),
        eval_mode=fedaf.get('eval_mode', 'SS'),
        global_epochs=fedaf.get('global_epochs', 500),
        global_lr=fedaf.get('global_lr', 0.001),
        server_batch_size=fedaf.get('server_batch_size', 256),
        loc_cdc=fedaf.get('loc_cdc', 0.0001),
        loc_lgkm=fedaf.get('loc_lgkm', 0.01),
        temperature=fedaf.get('temperature', 2.0),
        gamma=fedaf.get('gamma', 0.9),
        init=fedaf.get('init', 'real'),
        lr_img=lr_img,

        local_epochs=fedavg.get('local_epochs', 10),
        lr=fedavg.get('lr', 0.01),
        batch_size=fedavg.get('batch_size', 64),

        data_path=_resolve_path(paths.get('data', './data')),
        logits_dir=_resolve_path(paths.get('logits', './logits')),
        save_image_dir=_resolve_path(paths.get('images', './images')),
        save_path=_resolve_path(paths.get('result', './result')),
        models_root=models_root,
        plots_dir=_resolve_path(paths.get('plots', './plots')),
        model_dir=model_dir,

        plot_methods=list(plot.get('methods', ['fedaf', 'fedavg'])),
        plot_test_repeats=plot.get('test_repeats', 5),
    )

    # eval_it_pool: when periodic snapshots are on, sample start / middle / end
    # of condensation (matches the original [0, Iteration/2, Iteration] cadence).
    if cfg.eval_mode in ('S', 'SS'):
        step = max(1, cfg.Iteration // 2)
        cfg.eval_it_pool = list(range(0, cfg.Iteration + 1, step))
    else:
        cfg.eval_it_pool = [cfg.Iteration]

    return cfg


# --------------------------------------------------------------------------- #
#  Restart helpers (shared by both mains)                                      #
# --------------------------------------------------------------------------- #
def latest_completed_round(model_dir: str, method: str) -> int:
    """Highest round R such that <method>_global_model_R.pth exists (0 if none)."""
    if not os.path.isdir(model_dir):
        return 0
    prefix, suffix = f'{method}_global_model_', '.pth'
    best = 0
    for fn in os.listdir(model_dir):
        if fn.startswith(prefix) and fn.endswith(suffix):
            try:
                best = max(best, int(fn[len(prefix):-len(suffix)]))
            except ValueError:
                continue
    return best


def all_class_files_exist(path: str, prefix: str, num_classes: int) -> bool:
    """True if {prefix}kc_0..{C-1}.pt all exist under `path` (a finished phase)."""
    return os.path.isdir(path) and all(
        os.path.exists(os.path.join(path, f'{prefix}kc_{c}.pt'))
        for c in range(num_classes)
    )
