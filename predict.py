"""
Inference Entrypoint for GobbleCube ETA Challenge.

This module provides the `predict` function required by the grading harness. 
It loads the trained PyTorch neural network and executes a forward pass 
for a single trip request.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from features import build_features_scalar, get_feature_names

# --- Model Definition ---


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
        pu_emb = self.pu_embed(pz)
        do_emb = self.do_embed(dz)
        x = torch.cat([pu_emb, do_emb, cont, borough], dim=1)
        return self.net(x).squeeze(-1)


# --- Model Loading ---

_MODEL_PATH = Path(__file__).parent / "model.pkl"
with open(_MODEL_PATH, "rb") as _f:
    _bundle = pickle.load(_f)

_MODEL = ETAModel(**_bundle["config"])
_MODEL.load_state_dict(_bundle["state_dict"])
_MODEL.eval()

# --- Feature Index Precomputation ---

_FEATURE_NAMES = get_feature_names()
_BOROUGH_COLS = [i for i, n in enumerate(_FEATURE_NAMES) if n.startswith("borough_")]
_CONT_COLS = [
    "hour", "dow", "month", "passenger_count",
    "haversine_km", "bearing_deg", "log_haversine",
    "hour_sin", "hour_cos", "is_weekend", "is_rush_hour",
    "haversine_x_rush", "log_haversine_x_rush",
    "haversine_x_hour_sin", "haversine_x_hour_cos",
    "same_borough",
]
_CONT_INDICES = [_FEATURE_NAMES.index(c) for c in _CONT_COLS if c in _FEATURE_NAMES]


# --- Inference Interface ---

def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    ts = datetime.fromisoformat(request["requested_at"])
    pickup_zone = int(request["pickup_zone"])
    dropoff_zone = int(request["dropoff_zone"])

    row = build_features_scalar(
        pickup_zone=pickup_zone,
        dropoff_zone=dropoff_zone,
        hour=ts.hour,
        dow=ts.weekday(),
        month=ts.month,
        passenger_count=int(request["passenger_count"]),
    )[0]

    pz = torch.tensor([pickup_zone], dtype=torch.long)
    dz = torch.tensor([dropoff_zone], dtype=torch.long)
    cont = torch.tensor(row[_CONT_INDICES], dtype=torch.float32).unsqueeze(0)
    bor = torch.tensor(row[_BOROUGH_COLS], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        return float(_MODEL(pz, dz, cont, bor).item())