# server_fedaf.py  (corrected)
#
# Key fixes vs. original:
#   * LGKM now matches FULL per-class soft-label distributions (Eq. 13-16):
#       R = [r_0..r_{C-1}]  (client real-data soft labels, [C, C])  -- the target
#       T = [t_0..t_{C-1}]  (global model soft labels on synthetic data, [C, C])
#     instead of collapsing both to a [C] vector of diagonal confidences.
#   * Symmetric KL uses reduction='batchmean' (mathematically correct KL,
#     averaged over classes) rather than 'mean' (which was off by a factor C).
#   * Optimiser is SGD(lr, momentum=0.9) per the paper (momentum was missing).

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.utils_fedaf import load_latest_model, ensure_directory_exists, get_test_loader
import torch.optim as optim


def train_model(model, train_loader, R_matrix, num_classes, lambda_glob,
                temperature, device, num_epochs, lr=0.001, momentum=0.9):
    """Train the global model on the pooled synthetic data with CE + LGKM."""
    model.to(device)
    model.train()
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    eps = 1e-8
    R = R_matrix.to(device).clamp_min(eps)
    R = R / R.sum(dim=1, keepdim=True)          # [C, C], each row a distribution
    logR = R.log()

    for epoch in range(num_epochs):
        run_total = run_ce = run_lgkm = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)                          # [B, C]
            loss_ce = criterion_ce(outputs, labels)

            # --- build T = [t_c] over the classes present in this batch (Eq. 13) ---
            soft = F.softmax(outputs / temperature, dim=1)   # [B, C]
            present, t_rows = [], []
            for c in range(num_classes):
                mask = labels == c
                if mask.any():
                    t_rows.append(soft[mask].mean(dim=0))    # full [C] distribution
                    present.append(c)

            if t_rows:
                T = torch.stack(t_rows, dim=0).clamp_min(eps)        # [P, C]
                T = T / T.sum(dim=1, keepdim=True)
                Rp = R[present]                                      # [P, C]
                # LGKM = 1/2 ( KL(R||T) + KL(T||R) ), averaged over present classes (Eq. 15)
                kl_rt = F.kl_div(T.log(), Rp, reduction='batchmean')  # KL(R||T)
                kl_tr = F.kl_div(Rp.log(), T, reduction='batchmean')  # KL(T||R)
                loss_lgkm = 0.5 * (kl_rt + kl_tr)
            else:
                loss_lgkm = torch.zeros((), device=device)

            loss = loss_ce + lambda_glob * loss_lgkm
            loss.backward()
            optimizer.step()

            run_total += loss.item(); run_ce += loss_ce.item(); run_lgkm += float(loss_lgkm)

        n = max(1, len(train_loader))
        print(f"Server: Epoch {epoch + 1}/{num_epochs} | Total {run_total / n:.4f} "
              f"| CE {run_ce / n:.4f} | LGKM {run_lgkm / n:.4f}")


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted = torch.max(model(inputs).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy of the network on test images: {acc:.2f}%")
    return acc


def server_update(cfg, round_num):
    """Train the FedAF global model for `round_num` from pooled synthetic data.

    Every setting (dataset meta, paths, LGKM weight, epochs, LR, device) comes
    from the shared Config, so the server stays in lock-step with the clients
    and the plotter.
    """
    device = cfg.device
    num_classes = cfg.num_classes
    model_dir = cfg.model_dir
    ensure_directory_exists(model_dir)

    test_loader = get_test_loader(cfg, batch_size=cfg.server_batch_size)

    # ---- aggregated client soft labels R = [r_0..r_{C-1}], shape [C, C] ----
    global_R_path = os.path.join(cfg.logits_dir, 'Global', f'Round{round_num}_Global_Rc.pt')
    if os.path.exists(global_R_path):
        R_matrix = torch.load(global_R_path, map_location=device)
        if not (isinstance(R_matrix, torch.Tensor) and R_matrix.shape == (num_classes, num_classes)):
            raise ValueError(f"R has shape {getattr(R_matrix, 'shape', None)}; "
                             f"expected [{num_classes}, {num_classes}].")
        print(f"Server: loaded aggregated R matrix {tuple(R_matrix.shape)}.")
    else:
        print("Server: no aggregated R found; using uniform.")
        R_matrix = torch.full((num_classes, num_classes), 1.0 / num_classes, device=device)

    # ---- pool synthetic data from all clients ----
    all_images, all_labels = [], []
    for client_id in range(cfg.num_clients):
        fn = os.path.join(cfg.save_path, f'Client_{client_id}',
                          f'res_{cfg.method}_{cfg.dataset}_{cfg.model_name}_Client{client_id}_{cfg.ipc}ipc_Round{round_num}.pt')
        if os.path.exists(fn):
            d = torch.load(fn, map_location=device)
            if d.get('images') is not None and d['images'].size(0) > 0:
                all_images.append(d['images']); all_labels.append(d['labels'])
                print(f"Server: loaded synthetic data from Client {client_id}.")
        else:
            print(f"Server: no synthetic data for Client {client_id}. Skipping.")

    if not all_images:
        print("Server: no synthetic data aggregated. Skipping update.")
        return

    final_dataset = TensorDataset(torch.cat(all_images), torch.cat(all_labels))
    train_loader = DataLoader(final_dataset, batch_size=cfg.server_batch_size, shuffle=True)

    net = load_latest_model(model_dir, cfg.model_name, cfg.channel, num_classes, cfg.im_size, device)
    net.train()

    print("Server: training global model (CE + LGKM).")
    train_model(net, train_loader, R_matrix, num_classes, cfg.loc_lgkm,
                cfg.temperature, device, num_epochs=cfg.global_epochs, lr=cfg.global_lr)

    print("Server: evaluating global model.")
    evaluate_model(net, test_loader, device)

    model_path = os.path.join(model_dir, f"fedaf_global_model_{round_num}.pth")
    ensure_directory_exists(os.path.dirname(model_path))
    torch.save(net.state_dict(), model_path)
    print(f"Server: global model saved to {model_path}.")