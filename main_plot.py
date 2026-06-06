# main_plot.py
#
# Evaluates the saved per-round global models for each method and plots their
# test accuracy over communication rounds. Everything (dataset, model, client
# count, alpha, model/plot directories, methods, repeats) comes from config.yaml
# via config.load_config(), so the comparison always matches what was trained.

import os
import torch
import matplotlib
matplotlib.use('Agg')  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, set_start_method
import logging

from config import load_config
from utils.utils_fedaf import get_network, get_test_loader

logger = logging.getLogger(__name__)


def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test data and returns accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def evaluate_model_wrapper(task):
    method, model_file, round_num, cfg = task
    model_path = os.path.join(cfg.model_dir, model_file)

    if not os.path.exists(model_path):
        logger.warning(f"Model file does not exist: {model_path}")
        return method, round_num, None

    model = get_network(cfg.model, cfg.channel, cfg.num_classes, cfg.im_size).to(cfg.device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return method, round_num, None

    test_loader = get_test_loader(cfg)
    round_accuracies = [evaluate_model(model, test_loader, cfg.device)
                        for _ in range(cfg.plot_test_repeats)]
    avg_accuracy = float(np.mean(round_accuracies))
    logger.info(f"Method: {method}, Round {round_num}: Avg Test Accuracy = {avg_accuracy:.2f}%")
    return method, round_num, avg_accuracy


def test_saved_models(cfg):
    eval_tasks = []
    for method in cfg.plot_methods:
        if not os.path.isdir(cfg.model_dir):
            logger.warning(f"Directory does not exist: {cfg.model_dir}")
            continue
        model_files = sorted(
            [f for f in os.listdir(cfg.model_dir)
             if f.startswith(f"{method}_global_model") and f.endswith('.pth')],
            key=lambda x: int(x.split('_')[-1].split('.')[0]),
        )
        for model_file in model_files:
            try:
                round_num = int(model_file.split('_')[-1].split('.')[0])
            except (IndexError, ValueError):
                logger.error(f"Invalid model filename format: {model_file}")
                continue
            # Round 0 is the random init; skip it in the comparison plot.
            if round_num == 0:
                continue
            eval_tasks.append((method, model_file, round_num, cfg))

    method_accuracies = {method: {} for method in cfg.plot_methods}

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if eval_tasks:
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(evaluate_model_wrapper, eval_tasks)
        for method, round_num, avg_accuracy in results:
            if avg_accuracy is not None:
                method_accuracies[method][round_num] = avg_accuracy

    # ---- plot ----
    plt.figure(figsize=(12, 8))
    rounds = []
    for method, accuracies in method_accuracies.items():
        if not accuracies:
            logger.warning(f"No accuracies recorded for method: {method}. Skipping.")
            continue
        rounds = sorted(accuracies.keys())
        test_accuracies = [accuracies[r] for r in rounds]
        plt.plot(rounds, test_accuracies, marker='o', linestyle='-', label=method.upper())

    plt.title(f"Dataset: {cfg.dataset}, Users: {cfg.num_clients}, Alpha: {cfg.alpha}")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    if rounds:
        plt.xticks(rounds)
    plt.legend()

    os.makedirs(cfg.plots_dir, exist_ok=True)
    plot_save_path = os.path.join(
        cfg.plots_dir, f"{cfg.dataset}_{cfg.model}_C{cfg.num_clients}_alpha{cfg.alpha}.png")
    plt.savefig(plot_save_path)
    plt.close()
    logger.info(f"Plot saved to {plot_save_path}")
    print(f"Plot saved to {plot_save_path}")


def main():
    cfg = load_config('fedaf')
    os.makedirs(cfg.plots_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
        filename=os.path.join(cfg.plots_dir, 'plotting.log'),
        filemode='w',
    )
    test_saved_models(cfg)


if __name__ == "__main__":
    main()
