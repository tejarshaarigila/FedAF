# main_fedaf.py
#
# FedAF (Aggregation-Free Federated Learning) driver.
#
# All settings come from config.yaml via config.load_config('fedaf') -- there is
# no per-script ARGS class any more, so dataset / model / client-count / paths /
# hyperparameters are configured in exactly one place and shared with FedAvg and
# the plotter.
#
# Restartability (cfg.resume): a run resumes where it stopped. A round whose
# server model already exists is skipped; within a round each phase is skipped
# when its artifacts are present:
#   * Phase 1 (per-client V_{k,c})      -> logits/Client_*/Round_r/Vkc_*.pt
#   * Phase 1 aggregation               -> logits/Global/Round{r}_Global_Vc.pt
#   * Phase 2 condensation (per client) -> result/Client_*/res_*_Round{r}.pt
#   * Phase 2 soft labels (per client)  -> logits/Client_*/Round_r/Rkc_*.pt
#   * Phase 2 aggregation               -> logits/Global/Round{r}_Global_Rc.pt
#   * Server global-model update         -> models/.../fedaf_global_model_{r}.pth

import os
import time
import numpy as np
import torch
from torch.utils.data import Subset
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import load_config, Config, all_class_files_exist
from client.client_fedaf import Client
from server.server_fedaf import server_update
from utils.utils_fedaf import get_network, get_base_dataset, save_aggregated_logits


def initialize_global_model(cfg):
    model = get_network(cfg.model, cfg.channel, cfg.num_classes, cfg.im_size)
    os.makedirs(cfg.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.model_dir, 'fedaf_global_model_0.pth'))
    print("Global model initialized.")


def partition_indices(base_dataset, cfg):
    """Dirichlet label-skew partition of the dataset across cfg.num_clients."""
    labels = np.array(base_dataset.targets)
    indices = [[] for _ in range(cfg.num_clients)]
    for c in range(cfg.num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(np.repeat(cfg.alpha, cfg.num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for idx, split in enumerate(np.split(class_indices, proportions)):
            indices[idx].extend(split)
    return indices


def simulate(cfg=None):
    cfg = cfg or load_config('fedaf')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.ensure_dirs()

    print(f"[FedAF] dataset={cfg.dataset} model={cfg.model} K={cfg.num_clients} "
          f"alpha={cfg.alpha} rounds={cfg.rounds} ipc={cfg.ipc} device={cfg.device} "
          f"resume={cfg.resume}")

    base_dataset = get_base_dataset(cfg, train=True)
    data_partition_indices = partition_indices(base_dataset, cfg)

    if not os.path.exists(os.path.join(cfg.model_dir, 'fedaf_global_model_0.pth')):
        print("[+] Initializing Global Model")
        initialize_global_model(cfg)

    client_ids = list(range(cfg.num_clients))
    cfg_dict = cfg.to_dict()

    for r in range(1, cfg.rounds + 1):
        server_model_path = cfg.method_model_path('fedaf', r)
        if cfg.resume and os.path.exists(server_model_path):
            print(f"--- Round {r}/{cfg.rounds}: already complete, skipping. ---")
            continue
        print(f"--- Round {r}/{cfg.rounds} ---")

        # ---- Phase 1: clients compute v_{k,c} on REAL data with the clean model ----
        global_V_path = os.path.join(cfg.logits_dir, 'Global', f'Round{r}_Global_Vc.pt')
        if cfg.resume and os.path.exists(global_V_path):
            print(f"[resume] Phase 1 (V) already aggregated for round {r}.")
        else:
            _run_clients(client_compute_Vkc, client_ids, data_partition_indices,
                         cfg_dict, r, base_dataset, "client_compute_Vkc")
            save_aggregated_logits(aggregate_logits(cfg, 'V', r), cfg, r, 'V')

        # ---- Phase 2: condensation (+ re-sampling) and r_{k,c} on REAL data ----
        global_R_path = os.path.join(cfg.logits_dir, 'Global', f'Round{r}_Global_Rc.pt')
        if cfg.resume and os.path.exists(global_R_path):
            print(f"[resume] Phase 2 (R) already aggregated for round {r}.")
        else:
            _run_clients(client_data_condensation_and_Rkc, client_ids, data_partition_indices,
                         cfg_dict, r, base_dataset, "client_data_condensation_and_Rkc")
            save_aggregated_logits(aggregate_logits(cfg, 'R', r), cfg, r, 'R')

        # ---- Server trains the global model (CE + LGKM) ----
        server_update(cfg, r)


def _run_clients(fn, client_ids, data_partition_indices, cfg_dict, r, base_dataset, label):
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(fn, cid, data_partition_indices[cid], cfg_dict, r, base_dataset)
                for cid in client_ids]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"Error in {label}: {e}")


def client_compute_Vkc(client_id, data_partition_indices, cfg_dict, r, base_dataset):
    cfg = Config.from_dict(cfg_dict)
    vkc_path = os.path.join(cfg.logits_dir, f'Client_{client_id}', f'Round_{r}')
    if cfg.resume and all_class_files_exist(vkc_path, 'V', cfg.num_classes):
        print(f"[resume] Client {client_id}: Vkc already done for round {r}.")
        return
    client = Client(client_id, Subset(base_dataset, data_partition_indices), cfg)
    client.model = client.load_global_model()       # CLEAN global model (no resample)
    client.run_Vkc(r)


def client_data_condensation_and_Rkc(client_id, data_partition_indices, cfg_dict, r, base_dataset):
    cfg = Config.from_dict(cfg_dict)
    client = Client(client_id, Subset(base_dataset, data_partition_indices), cfg)
    client.model = client.load_global_model()       # CLEAN global model (kept for R)

    global_Vc_path = os.path.join(cfg.logits_dir, 'Global', f'Round{r}_Global_Vc.pt')
    while not os.path.exists(global_Vc_path):
        time.sleep(1)

    # Condensation (re-samples per step internally) -- skip if already saved.
    if cfg.resume and os.path.exists(client.synthetic_save_path(r)):
        print(f"[resume] Client {client_id}: synthetic data already done for round {r}.")
    else:
        client.run_data_condensation(r)

    # Soft labels r_{k,c} from real data + clean model -- skip if already saved.
    rkc_path = os.path.join(cfg.logits_dir, f'Client_{client_id}', f'Round_{r}')
    if cfg.resume and all_class_files_exist(rkc_path, 'R', cfg.num_classes):
        print(f"[resume] Client {client_id}: Rkc already done for round {r}.")
    else:
        client.run_Rkc(r)


def aggregate_logits(cfg, v_r, r):
    """Aggregate per-client class statistics into a [C, C] matrix.

    Each row c is the mean of v_{k,c} (or r_{k,c}) over the clients that own
    class c (count > 0). Classes owned by nobody stay as a zero row (for R the
    server clamps + renormalises a zero row to uniform).
    """
    num_classes = cfg.num_classes
    agg = torch.zeros(num_classes, num_classes, device=cfg.device)
    cnt = torch.zeros(num_classes, device=cfg.device)
    for cid in range(cfg.num_clients):
        base = os.path.join(cfg.logits_dir, f'Client_{cid}', f'Round_{r}')
        for c in range(num_classes):
            p = os.path.join(base, f'{v_r}kc_{c}.pt')
            if not os.path.exists(p):
                continue
            try:
                d = torch.load(p, map_location=cfg.device)
                value = d['value'] if isinstance(d, dict) else d
                count = (d.get('count', 1) if isinstance(d, dict) else 1)
                if isinstance(value, torch.Tensor) and value.numel() == num_classes and count > 0:
                    agg[c] += value.to(cfg.device).view(-1)
                    cnt[c] += 1
            except Exception as e:
                print(f"Server: failed to load {v_r}kc for client {cid}, class {c} - {e}")
    denom = torch.where(cnt == 0, torch.ones_like(cnt), cnt).unsqueeze(1)
    M = agg / denom
    print(f"Server: aggregated {v_r} matrix {tuple(M.shape)} "
          f"({int((cnt > 0).sum())}/{num_classes} classes populated).")
    return M


if __name__ == '__main__':
    simulate()
