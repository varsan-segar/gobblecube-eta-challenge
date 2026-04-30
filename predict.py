"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from features import build_features_scalar, get_feature_names

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)
# Disable xgboost's feature-name validation so we can predict on a bare
# numpy array (skips per-call DataFrame construction overhead).
if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None

_FEATURE_NAMES = get_feature_names()


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
    x = build_features_scalar(
        pickup_zone=int(request["pickup_zone"]),
        dropoff_zone=int(request["dropoff_zone"]),
        hour=ts.hour,
        dow=ts.weekday(),
        month=ts.month,
        passenger_count=int(request["passenger_count"]),
    )
    return float(_MODEL.predict(x)[0])