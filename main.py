#!/usr/bin/env python3
"""
EL9123 (Intro ML) Class Project Script
=====================================

Project: PCA + MLP vs CNN for Object Localization (Bounding Box Regression)
Type: Regression (predict bbox parameters)
Domain: Computer Vision

What this script does
---------------------
1) Builds a synthetic "object detection" dataset by pasting MNIST digits onto a larger canvas.
   - Input X: 1xHxW grayscale image (e.g., 64x64)
   - Output y: bounding box (xc, yc, w, h) normalized to [0, 1]

2) Trains TWO models:
   A) PCA + MLP (classic ML flavor, builds on PCA labs)
   B) CNN regressor (end-to-end baseline)

3) Evaluates with:
   - MSE on bbox parameters
   - Mean IoU (Intersection-over-Union) of predicted vs GT boxes

4) Produces visualizations:
   - Scatter plot: GT vs Pred (xc, yc) for the test set
   - Qualitative examples: predicted boxes overlaid on images (saved as PNG)

Dependencies
------------
pip install numpy matplotlib scikit-learn torch torchvision

Run
---
python project_localization_pca_cnn.py --epochs 8 --device cpu
python project_localization_pca_cnn.py --epochs 15 --device cuda

Notes
-----
- This is intentionally a "simplified detection" problem framed as regression.
- It's a strong next-step from PCA+NN and CNN labs: the output is no longer a class label,
  but a structured prediction (bbox).
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms


# ---------------------------
# Reproducibility utilities
# ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: For strict determinism (may slow down):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Bounding box utilities
# ---------------------------

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def bbox_xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    """
    Convert normalized bbox (xc, yc, w, h) -> (x1, y1, x2, y2), all normalized.
    b shape: (..., 4)
    """
    xc, yc, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    out = np.stack([x1, y1, x2, y2], axis=-1)
    return clamp01(out)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    IoU for normalized boxes in (x1,y1,x2,y2).
    a, b shape: (N, 4)
    returns: (N,)
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - inter_area + 1e-9
    return inter_area / union


# ---------------------------
# Synthetic dataset creation
# ---------------------------

@dataclass
class SyntheticConfig:
    canvas_size: int = 64
    digit_scale_min: float = 0.6
    digit_scale_max: float = 1.2
    noise_std: float = 0.05
    max_translate: int = 8  # additional margin; placement is randomized across the canvas


def paste_digit_on_canvas(
    digit: np.ndarray,
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pasts a single MNIST digit onto a larger canvas at a random location and scale.

    Args:
      digit: (28, 28) float in [0,1]
    Returns:
      canvas: (H, W) float in [0,1]
      bbox: (4,) normalized (xc, yc, w, h)
    """
    H = W = cfg.canvas_size
    canvas = np.zeros((H, W), dtype=np.float32)

    # Random scale
    scale = rng.uniform(cfg.digit_scale_min, cfg.digit_scale_max)
    new_size = int(round(28 * scale))
    new_size = max(12, min(new_size, H - 2))  # keep sane

    # Resize digit using simple nearest-neighbor (fast, good enough for class project)
    # For higher fidelity, you could use cv2 or PIL, but keep dependencies minimal.
    yy = (np.linspace(0, 27, new_size)).astype(np.int32)
    xx = (np.linspace(0, 27, new_size)).astype(np.int32)
    digit_resized = digit[np.ix_(yy, xx)]

    dh, dw = digit_resized.shape

    # Random position
    top = rng.integers(0, H - dh)
    left = rng.integers(0, W - dw)

    # Paste (max to keep foreground bright)
    canvas[top:top + dh, left:left + dw] = np.maximum(canvas[top:top + dh, left:left + dw], digit_resized)

    # Add mild gaussian noise
    if cfg.noise_std > 0:
        canvas = canvas + rng.normal(0.0, cfg.noise_std, size=canvas.shape).astype(np.float32)

    canvas = np.clip(canvas, 0.0, 1.0)

    # BBox in pixel coords
    x1, y1 = left, top
    x2, y2 = left + dw, top + dh

    # Normalize to [0,1]
    xc = ((x1 + x2) / 2.0) / W
    yc = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    bbox = np.array([xc, yc, bw, bh], dtype=np.float32)

    return canvas, bbox


def load_data(
    data_dir: str,
    n_samples: int,
    cfg: SyntheticConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder-style data loader: assumes dataset is available locally.
    Here we use torchvision's MNIST (downloaded/cached under data_dir),
    then generate a NEW dataset by composing digits onto a canvas.

    Returns:
      X: (N, 1, H, W) float32
      y: (N, 4) float32 (xc,yc,w,h) normalized
    """
    rng = np.random.default_rng(seed)

    # MNIST will download if missing (still "local" once cached).
    mnist = MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())

    X = np.zeros((n_samples, 1, cfg.canvas_size, cfg.canvas_size), dtype=np.float32)
    y = np.zeros((n_samples, 4), dtype=np.float32)

    for i in range(n_samples):
        idx = rng.integers(0, len(mnist))
        img_tensor, _label = mnist[idx]
        digit = img_tensor.squeeze(0).numpy().astype(np.float32)  # (28,28) in [0,1]

        canvas, bbox = paste_digit_on_canvas(digit, cfg, rng)
        X[i, 0] = canvas
        y[i] = bbox

    return X, y


# ---------------------------
# PyTorch datasets
# ---------------------------

class BBoxDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx]).float()     # (1,H,W)
        y = torch.from_numpy(self.y[idx]).float()     # (4,)
        return x, y


class PCADataset(Dataset):
    """
    Dataset that returns PCA features (already computed) and y.
    """
    def __init__(self, Z: np.ndarray, y: np.ndarray):
        super().__init__()
        self.Z = Z
        self.y = y

    def __len__(self) -> int:
        return int(self.Z.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.from_numpy(self.Z[idx]).float()     # (D,)
        y = torch.from_numpy(self.y[idx]).float()     # (4,)
        return z, y


# ---------------------------
# Models
# ---------------------------

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),  # bbox params are normalized to [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNRegressor(nn.Module):
    """
    A small CNN regression model for bbox (xc,yc,w,h).
    Designed to be stable for a class project and run fast on CPU.
    """
    def __init__(self, img_size: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # 64 -> 32
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid(),  # normalized bbox outputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


# ---------------------------
# Training & evaluation
# ---------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        running += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return running / max(1, n)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        gt = yb.numpy()

        preds.append(pred)
        gts.append(gt)

    P = np.concatenate(preds, axis=0)
    G = np.concatenate(gts, axis=0)

    mse = float(np.mean((P - G) ** 2))

    P_xyxy = bbox_xywh_to_xyxy(P)
    G_xyxy = bbox_xywh_to_xyxy(G)
    mean_iou = float(np.mean(iou_xyxy(P_xyxy, G_xyxy)))

    return {"mse": mse, "mean_iou": mean_iou}


# ---------------------------
# Visualization
# ---------------------------

def save_scatter_gt_vs_pred(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    """
    Scatter plot for (xc, yc): GT vs Pred
    """
    plt.figure()
    plt.scatter(gt[:, 0], pred[:, 0], marker='o', alpha=0.4, label='xc')
    plt.scatter(gt[:, 1], pred[:, 1], marker='x', alpha=0.4, label='yc')
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def draw_bbox_on_ax(ax, bbox_xyxy_norm: np.ndarray, H: int, W: int, label: str) -> None:
    """
    Draw bbox given normalized (x1,y1,x2,y2) onto matplotlib axis.
    """
    x1, y1, x2, y2 = bbox_xyxy_norm
    x1p, y1p = x1 * W, y1 * H
    wp, hp = (x2 - x1) * W, (y2 - y1) * H

    rect = plt.Rectangle((x1p, y1p), wp, hp, fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.text(x1p, max(0, y1p - 2), label, fontsize=9, va="bottom")


@torch.no_grad()
def save_qualitative_examples(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    out_path: str,
    n: int = 12,
    title: str = "",
) -> None:
    """
    Saves a grid of images with GT and Pred boxes overlaid.
    """
    model.eval()
    idxs = np.random.choice(len(X), size=min(n, len(X)), replace=False)

    ncols = 4
    nrows = int(np.ceil(len(idxs) / ncols))
    H, W = X.shape[2], X.shape[3]

    plt.figure(figsize=(12, 3 * nrows))
    for i, idx in enumerate(idxs, start=1):
        img = X[idx, 0]
        gt = y[idx]

        xb = torch.from_numpy(X[idx:idx + 1]).float().to(device)
        pred = model(xb).cpu().numpy()[0]

        gt_xyxy = bbox_xywh_to_xyxy(gt[None, :])[0]
        pr_xyxy = bbox_xywh_to_xyxy(pred[None, :])[0]

        ax = plt.subplot(nrows, ncols, i)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        draw_bbox_on_ax(ax, gt_xyxy, H, W, "GT")
        draw_bbox_on_ax(ax, pr_xyxy, H, W, "Pred")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ---------------------------
# PCA pipeline
# ---------------------------

def fit_pca_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, PCA, StandardScaler]:
    """
    Flatten images, standardize, fit PCA on train only, transform both sets.
    """
    Ntr = X_train.shape[0]
    Nts = X_test.shape[0]
    D = int(np.prod(X_train.shape[1:]))

    Xtr_flat = X_train.reshape(Ntr, D)
    Xts_flat = X_test.reshape(Nts, D)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_scaled = scaler.fit_transform(Xtr_flat)
    Xts_scaled = scaler.transform(Xts_flat)

    pca = PCA(n_components=n_components, random_state=0)
    Ztr = pca.fit_transform(Xtr_scaled).astype(np.float32)
    Zts = pca.transform(Xts_scaled).astype(np.float32)

    return Ztr, Zts, pca, scaler


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Where to store/load MNIST.")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Where to save plots/results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--n_samples", type=int, default=12000)
    parser.add_argument("--test_size", type=float, default=0.2)

    # PCA + MLP hyperparams
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)

    # CNN hyperparams
    parser.add_argument("--cnn_lr", type=float, default=1e-3)

    # Training common
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[Info] Using device: {device}")

    # 1) Data loading & preprocessing
    cfg = SyntheticConfig(canvas_size=64, noise_std=0.05)
    print("[Info] Loading and generating synthetic localization dataset...")
    X, y = load_data(args.data_dir, args.n_samples, cfg, seed=args.seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"[Info] Train: {X_train.shape}, Test: {X_test.shape}")

    # 2) PCA features for MLP model
    print("[Info] Fitting PCA (train only) and transforming features...")
    Z_train, Z_test, pca, scaler = fit_pca_features(X_train, X_test, n_components=args.pca_components)
    print(f"[Info] PCA explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

    # 3) Build DataLoaders
    pca_train_ds = PCADataset(Z_train, y_train)
    pca_test_ds = PCADataset(Z_test, y_test)
    pca_train_loader = DataLoader(pca_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    pca_test_loader = DataLoader(pca_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cnn_train_ds = BBoxDataset(X_train, y_train)
    cnn_test_ds = BBoxDataset(X_test, y_test)
    cnn_train_loader = DataLoader(cnn_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    cnn_test_loader = DataLoader(cnn_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 4) Define models
    loss_fn = nn.MSELoss()

    mlp = MLPRegressor(in_dim=args.pca_components, hidden_dim=args.mlp_hidden, dropout=0.1).to(device)
    cnn = CNNRegressor(img_size=cfg.canvas_size).to(device)

    mlp_opt = optim.Adam(mlp.parameters(), lr=args.mlp_lr)
    cnn_opt = optim.Adam(cnn.parameters(), lr=args.cnn_lr)

    # 5) Training loops
    print("\n[Train] PCA + MLP regressor")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(mlp, pca_train_loader, mlp_opt, device, loss_fn)
        metrics = evaluate(mlp, pca_test_loader, device)
        print(f"  Epoch {epoch:02d}/{args.epochs} | train_loss={tr_loss:.6f} | test_mse={metrics['mse']:.6f} | mean_iou={metrics['mean_iou']:.4f}")

    print("\n[Train] CNN regressor")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(cnn, cnn_train_loader, cnn_opt, device, loss_fn)
        metrics = evaluate(cnn, cnn_test_loader, device)
        print(f"  Epoch {epoch:02d}/{args.epochs} | train_loss={tr_loss:.6f} | test_mse={metrics['mse']:.6f} | mean_iou={metrics['mean_iou']:.4f}")

    # 6) Final evaluation + visualizations
    print("\n[Eval] Final evaluation on test set...")
    mlp_metrics = evaluate(mlp, pca_test_loader, device)
    cnn_metrics = evaluate(cnn, cnn_test_loader, device)
    print(f"  PCA+MLP: test_mse={mlp_metrics['mse']:.6f}, mean_iou={mlp_metrics['mean_iou']:.4f}")
    print(f"  CNN    : test_mse={cnn_metrics['mse']:.6f}, mean_iou={cnn_metrics['mean_iou']:.4f}")

    # Get predictions for scatter plots (xc, yc)
    @torch.no_grad()
    def collect_preds(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        preds, gts = [], []
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            gts.append(yb.numpy())
        return np.concatenate(gts, axis=0), np.concatenate(preds, axis=0)

    gt_mlp, pr_mlp = collect_preds(mlp, pca_test_loader)
    gt_cnn, pr_cnn = collect_preds(cnn, cnn_test_loader)

    scatter_mlp_path = os.path.join(args.out_dir, "scatter_gt_vs_pred_pca_mlp.png")
    scatter_cnn_path = os.path.join(args.out_dir, "scatter_gt_vs_pred_cnn.png")
    save_scatter_gt_vs_pred(gt_mlp, pr_mlp, scatter_mlp_path, "PCA+MLP: GT vs Pred (xc, yc)")
    save_scatter_gt_vs_pred(gt_cnn, pr_cnn, scatter_cnn_path, "CNN: GT vs Pred (xc, yc)")

    # Qualitative examples
    qual_mlp_path = os.path.join(args.out_dir, "qualitative_pca_mlp.png")
    qual_cnn_path = os.path.join(args.out_dir, "qualitative_cnn.png")
    save_qualitative_examples(
        mlp, X_test, y_test, device, qual_mlp_path, n=12,
        title=f"PCA+MLP qualitative (mean IoU={mlp_metrics['mean_iou']:.3f})"
    )
    save_qualitative_examples(
        cnn, X_test, y_test, device, qual_cnn_path, n=12,
        title=f"CNN qualitative (mean IoU={cnn_metrics['mean_iou']:.3f})"
    )

    # Save a brief results summary
    summary_path = os.path.join(args.out_dir, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Results Summary\n")
        f.write("================\n")
        f.write(f"Dataset: Synthetic MNIST-on-canvas localization\n")
        f.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")
        f.write(f"PCA components: {args.pca_components}, explained_var_sum={pca.explained_variance_ratio_.sum():.4f}\n\n")
        f.write(f"PCA+MLP: test_mse={mlp_metrics['mse']:.6f}, mean_iou={mlp_metrics['mean_iou']:.4f}\n")
        f.write(f"CNN    : test_mse={cnn_metrics['mse']:.6f}, mean_iou={cnn_metrics['mean_iou']:.4f}\n")

    print("\n[Done] Saved outputs to:")
    print(f"  - {args.out_dir}")
    print("Key files:")
    print(f"  - {os.path.basename(scatter_mlp_path)}")
    print(f"  - {os.path.basename(scatter_cnn_path)}")
    print(f"  - {os.path.basename(qual_mlp_path)}")
    print(f"  - {os.path.basename(qual_cnn_path)}")
    print(f"  - {os.path.basename(summary_path)}")


if __name__ == "__main__":
    main()
