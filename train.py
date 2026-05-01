#!/usr/bin/env python
"""
train.py: Neural Network Training Script

This script trains a PyTorch MLP with zone embeddings for NYC taxi trip duration prediction.
It loads data, builds features, trains the model, and saves the final 'model.pkl' to the root directory.

Architecture:
    - Zone embeddings: 265 zones → 64-dim embedding
    - Numeric features: Continuous temporal/geographic + borough flags
    - MLP: [256, 128, 64] with BatchNorm and ReLU
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from features import build_features

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "model.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
EMBED_DIM = 64
HIDDEN = [256, 128, 64]
BATCH_SIZE = 8192
EPOCHS = 30
LR = 3e-4
PATIENCE = 5
NUM_ZONES = 266  # 1-265 + padding 0


class ETAModel(nn.Module):
    """Zone embeddings + MLP for trip duration prediction."""

    def __init__(self, num_zones: int, embed_dim: int, num_cont: int,
                 num_borough: int, hidden: list[int]):
        super().__init__()
        self.pu_embed = nn.Embedding(num_zones, embed_dim)
        self.do_embed = nn.Embedding(num_zones, embed_dim)

        total_in = 2 * embed_dim + num_cont + num_borough

        layers = []
        prev = total_in
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, pz: torch.Tensor, dz: torch.Tensor,
                cont: torch.Tensor, borough: torch.Tensor) -> torch.Tensor:
        """Forward pass concatenating embeddings with continuous features."""
        pu_emb = self.pu_embed(pz)
        do_emb = self.do_embed(dz)
        x = torch.cat([pu_emb, do_emb, cont, borough], dim=1)
        return self.net(x).squeeze(-1)


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features into embeddings (zone IDs), continuous, and borough flags."""
    feat_df = build_features(df)

    pz = feat_df["pickup_zone"].values.astype(np.int64)
    dz = feat_df["dropoff_zone"].values.astype(np.int64)

    cont_cols = [
        "hour", "dow", "month", "passenger_count",
        "haversine_km", "bearing_deg", "log_haversine",
        "hour_sin", "hour_cos", "is_weekend", "is_rush_hour",
        "haversine_x_rush", "log_haversine_x_rush",
        "haversine_x_hour_sin", "haversine_x_hour_cos",
        "same_borough",
    ]
    cont_cols_actual = [c for c in cont_cols if c in feat_df.columns]
    cont = feat_df[cont_cols_actual].values.astype(np.float32)

    borough_cols = [c for c in feat_df.columns if c.startswith("borough_")]
    bor = feat_df[borough_cols].values.astype(np.float32)

    return pz, dz, cont, bor


def train_model(model: nn.Module, loader: DataLoader, val_loader: DataLoader) -> float:
    """Train the model with early stopping. Returns best validation MAE."""
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.L1Loss()

    best_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"    epoch {epoch+1:3d}", unit="batch", leave=False)
        
        for pz, dz, cont, bor, y in pbar:
            pz = pz.to(DEVICE, non_blocking=True)
            dz = dz.to(DEVICE, non_blocking=True)
            cont = cont.to(DEVICE, non_blocking=True)
            bor = bor.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            preds = model(pz, dz, cont, bor)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.1f}")

        pbar.close()

        # Validation phase
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for pz, dz, cont, bor, y in val_loader:
                pz = pz.to(DEVICE, non_blocking=True)
                dz = dz.to(DEVICE, non_blocking=True)
                cont = cont.to(DEVICE, non_blocking=True)
                bor = bor.to(DEVICE, non_blocking=True)
                val_preds.append(model(pz, dz, cont, bor).cpu().numpy())
                val_true.append(y.numpy())

        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_mae = float(np.mean(np.abs(val_preds - val_true)))

        scheduler.step(val_mae)

        print(f"    epoch {epoch+1:3d}  train_loss: {total_loss/n_batches:.1f}  "
              f"val_mae: {val_mae:.1f}s  lr: {scheduler.get_last_lr()[0]:.1e}", flush=True)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_mae


def main() -> None:
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    
    if not train_path.exists() or not dev_path.exists():
        print("Missing data files. Run: python data/download_data.py")
        sys.exit(1)

    print("\nLoading data ...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,}  dev: {len(dev):,}")

    print("Building features ...")
    t0 = time.time()
    pz_tr, dz_tr, cont_tr, bor_tr = _prepare_features(train)
    y_tr = train["duration_seconds"].to_numpy().astype(np.float32)
    
    pz_dv, dz_dv, cont_dv, bor_dv = _prepare_features(dev)
    y_dv = dev["duration_seconds"].to_numpy().astype(np.float32)
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  continuous dims: {cont_tr.shape[1]}  |  borough dims: {bor_tr.shape[1]}")

    # Create 10% validation split from training data
    n = len(pz_tr)
    idx = np.random.RandomState(42).permutation(n)
    n_val = n // 10
    tr_idx, val_idx = idx[n_val:], idx[:n_val]

    tr_ds = TensorDataset(
        torch.tensor(pz_tr[tr_idx]), torch.tensor(dz_tr[tr_idx]),
        torch.tensor(cont_tr[tr_idx]), torch.tensor(bor_tr[tr_idx]),
        torch.tensor(y_tr[tr_idx]),
    )
    val_ds = TensorDataset(
        torch.tensor(pz_tr[val_idx]), torch.tensor(dz_tr[val_idx]),
        torch.tensor(cont_tr[val_idx]), torch.tensor(bor_tr[val_idx]),
        torch.tensor(y_tr[val_idx]),
    )
    
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, prefetch_factor=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                            num_workers=4, prefetch_factor=2, pin_memory=True)

    print(f"\nTraining NN ({len(HIDDEN)} hidden layers, embed_dim={EMBED_DIM})")
    print(f"  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}  patience={PATIENCE}\n")

    model = ETAModel(NUM_ZONES, EMBED_DIM, cont_tr.shape[1], bor_tr.shape[1], HIDDEN)
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}\n")

    t0 = time.time()
    best_val_mae = train_model(model, tr_loader, val_loader)
    print(f"\n  trained in {time.time() - t0:.0f}s  |  best val MAE: {best_val_mae:.1f}s")

    # Evaluate against the holdout dev set
    model = model.to(DEVICE)
    model.eval()
    dv_ds = TensorDataset(
        torch.tensor(pz_dv), torch.tensor(dz_dv),
        torch.tensor(cont_dv), torch.tensor(bor_dv),
    )
    dv_loader = DataLoader(dv_ds, batch_size=BATCH_SIZE * 2)

    preds = []
    with torch.no_grad():
        for pz, dz, cont, bor in dv_loader:
            pz = pz.to(DEVICE, non_blocking=True)
            dz = dz.to(DEVICE, non_blocking=True)
            cont = cont.to(DEVICE, non_blocking=True)
            bor = bor.to(DEVICE, non_blocking=True)
            preds.append(model(pz, dz, cont, bor).cpu().numpy())

    preds = np.concatenate(preds)
    dev_mae = float(np.mean(np.abs(preds - y_dv)))

    print(f"\n{'─'*55}")
    print(f"  Dev MAE: {dev_mae:.1f}s")
    print(f"{'─'*55}")

    # Save final model state and configuration to root
    model_cpu = model.cpu()
    model_cpu.eval()

    model_bundle = {
        "type": "nn",
        "state_dict": model_cpu.state_dict(),
        "config": {
            "num_zones": NUM_ZONES,
            "embed_dim": EMBED_DIM,
            "num_cont": cont_tr.shape[1],
            "num_borough": bor_tr.shape[1],
            "hidden": HIDDEN,
        },
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\n  model saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()