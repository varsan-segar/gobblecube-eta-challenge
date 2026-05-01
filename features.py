"""Shared feature engineering — imported by experiments, train.py, and predict.py.

Features:
    Baseline: pickup_zone, dropoff_zone, hour, dow, month, passenger_count
    Zone geometry: haversine_km, bearing_deg, pickup_borough, dropoff_borough
    Zone-pair interaction: same_borough flag
    Temporal: hour_sin, hour_cos (cyclic), is_weekend, is_rush_hour
    Interactions: distance × rush hour, distance × time-of-day (hour_sin/cos)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Zone centroid lookup (lazy-loaded from zone_centroids.csv)
# ---------------------------------------------------------------------------
_CENTROIDS_PATH = Path(__file__).resolve().parent / "data" / "zone_centroids.csv"

# Caches: zone_id -> (lat, lon, borough)
_zone_lat: Dict[int, float] = {}
_zone_lon: Dict[int, float] = {}
_zone_borough: Dict[int, str] = {}
_borough_to_idx: Dict[str, int] = {}
_loaded = False

# Precomputed haversine lookup: (pickup_id, dropoff_id) -> distance_km
_distance_cache: Dict[Tuple[int, int], float] = {}


def _load_centroids() -> None:
    """Load zone centroids CSV into module-level dicts (called once)."""
    global _loaded, _borough_to_idx
    if _loaded:
        return
    df = pd.read_csv(_CENTROIDS_PATH)
    for row in df.itertuples(index=False):
        zid = int(row.LocationID)
        _zone_lat[zid] = float(row.lat)
        _zone_lon[zid] = float(row.lon)
        _zone_borough[zid] = str(row.borough)
    # Build borough -> 0-based index mapping
    _borough_to_idx = {b: i for i, b in enumerate(sorted(set(_zone_borough.values())))}
    _loaded = True


# ---------------------------------------------------------------------------
# Vectorized haversine (numpy)
# ---------------------------------------------------------------------------
def _haversine_vec(lat1: np.ndarray, lon1: np.ndarray,
                   lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in kilometers."""
    r = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing_vec(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized initial bearing in degrees [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return np.mod(bearing, 360.0)


# ---------------------------------------------------------------------------
# Numeric feature names (ordered list for model consistency)
# ---------------------------------------------------------------------------
# Baseline
NUMERIC_FEATURES: List[str] = [
    "pickup_zone",
    "dropoff_zone",
    "hour",
    "dow",
    "month",
    "passenger_count",
]
# Zone geometry
GEO_FEATURES: List[str] = [
    "haversine_km",
    "bearing_deg",
    "log_haversine",
]
# Temporal
TEMP_FEATURES: List[str] = [
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_rush_hour",
]
# Interaction features (distance × time)
INTERACTION_FEATURES: List[str] = [
    "haversine_x_rush",
    "log_haversine_x_rush",
    "haversine_x_hour_sin",
    "haversine_x_hour_cos",
]
# Borough features (one-hot, dynamically named)
# Generated at runtime from NUM_BOROUGHS

ALL_NUMERIC_FEATURES: List[str] = (
    NUMERIC_FEATURES + GEO_FEATURES + TEMP_FEATURES + INTERACTION_FEATURES
)

NUM_BOROUGHS: int = 0  # Set after _load_centroids


def get_feature_names() -> List[str]:
    """Return final ordered feature list (numeric + borough one-hot)."""
    _load_centroids()
    global NUM_BOROUGHS
    if NUM_BOROUGHS == 0:
        NUM_BOROUGHS = len(_borough_to_idx)
    borough_cols = [f"borough_pu_{i}" for i in range(NUM_BOROUGHS)]
    borough_cols += [f"borough_do_{i}" for i in range(NUM_BOROUGHS)]
    return ALL_NUMERIC_FEATURES + ["same_borough"] + borough_cols


@np.vectorize
def _get_haversine_single(pu: int, do: int) -> float:
    """Single-pair haversine (cached). Used by predict() scalar codepath."""
    key = (pu, do)
    if key in _distance_cache:
        return _distance_cache[key]
    _load_centroids()
    lat1, lon1 = _zone_lat.get(pu, 40.7), _zone_lon.get(pu, -73.9)
    lat2, lon2 = _zone_lat.get(do, 40.7), _zone_lon.get(do, -73.9)
    d = float(_haversine_vec(np.array([lat1]), np.array([lon1]),
                              np.array([lat2]), np.array([lon2]))[0])
    _distance_cache[key] = d
    return d


# ---------------------------------------------------------------------------
# DataFrame feature builder (train/dev codepath)
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build full feature matrix from a raw trip DataFrame.

    Expected columns: pickup_zone, dropoff_zone, requested_at, passenger_count.

    Returns DataFrame with columns matching get_feature_names().
    """
    _load_centroids()
    global NUM_BOROUGHS
    if NUM_BOROUGHS == 0:
        NUM_BOROUGHS = len(_borough_to_idx)

    ts = pd.to_datetime(df["requested_at"])
    hour = ts.dt.hour.astype("int8")
    dow = ts.dt.dayofweek.astype("int8")
    month = ts.dt.month.astype("int8")
    pax = df["passenger_count"].astype("int8")
    pu = df["pickup_zone"].astype("int32")
    do = df["dropoff_zone"].astype("int32")

    # Map centroids
    pu_lat = pu.map(_zone_lat).fillna(40.7).to_numpy()
    pu_lon = pu.map(_zone_lon).fillna(-73.9).to_numpy()
    do_lat = do.map(_zone_lat).fillna(40.7).to_numpy()
    do_lon = do.map(_zone_lon).fillna(-73.9).to_numpy()

    haversine = _haversine_vec(pu_lat, pu_lon, do_lat, do_lon).astype("float32")
    bearing = _bearing_vec(pu_lat, pu_lon, do_lat, do_lon).astype("float32")
    log_hav = np.log1p(haversine).astype("float32")
    hour_sin = np.sin(2 * np.pi * hour / 24).astype("float32")
    hour_cos = np.cos(2 * np.pi * hour / 24).astype("float32")
    is_weekend = (dow >= 5).astype("int8")
    is_rush = hour.isin([7, 8, 9, 17, 18, 19]).astype("float32")

    features = pd.DataFrame(
        {
            "pickup_zone": pu,
            "dropoff_zone": do,
            "hour": hour,
            "dow": dow,
            "month": month,
            "passenger_count": pax,
            "haversine_km": haversine,
            "bearing_deg": bearing,
            "log_haversine": log_hav,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush,
            # Interaction features (distance × time-of-day)
            "haversine_x_rush": (haversine * is_rush).astype("float32"),
            "log_haversine_x_rush": (log_hav * is_rush).astype("float32"),
            "haversine_x_hour_sin": (haversine * hour_sin).astype("float32"),
            "haversine_x_hour_cos": (haversine * hour_cos).astype("float32"),
        }
    )

    # Borough features
    pu_borough = pu.map(_zone_borough).fillna("Unknown")
    do_borough = do.map(_zone_borough).fillna("Unknown")
    features["same_borough"] = (pu_borough == do_borough).astype("int8")

    for bname, bidx in _borough_to_idx.items():
        features[f"borough_pu_{bidx}"] = (pu_borough == bname).astype("int8")
        features[f"borough_do_{bidx}"] = (do_borough == bname).astype("int8")

    # Ensure column order
    return features[get_feature_names()]


# ---------------------------------------------------------------------------
# Scalar feature builder (used by predict.py for single request)
# ---------------------------------------------------------------------------
def build_features_scalar(pickup_zone: int, dropoff_zone: int,
                          hour: int, dow: int, month: int,
                          passenger_count: int) -> np.ndarray:
    """Build a single row feature vector (1, N) for inference.

    Returns numpy float32 array, shape (1, len(get_feature_names())).
    """
    _load_centroids()
    global NUM_BOROUGHS
    if NUM_BOROUGHS == 0:
        NUM_BOROUGHS = len(_borough_to_idx)

    haversine = _get_haversine_single(pickup_zone, dropoff_zone)
    bearing = float(_bearing_vec(
        np.array([_zone_lat.get(pickup_zone, 40.7)]),
        np.array([_zone_lon.get(pickup_zone, -73.9)]),
        np.array([_zone_lat.get(dropoff_zone, 40.7)]),
        np.array([_zone_lon.get(dropoff_zone, -73.9)]),
    )[0])

    log_hav = float(np.log1p(haversine))
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    is_weekend = 1.0 if dow >= 5 else 0.0
    is_rush = 1.0 if hour in (7, 8, 9, 17, 18, 19) else 0.0

    # Interaction features
    haversine_x_rush = haversine * is_rush
    log_haversine_x_rush = log_hav * is_rush
    haversine_x_hour_sin = haversine * hour_sin
    haversine_x_hour_cos = haversine * hour_cos

    pu_borough = _zone_borough.get(pickup_zone, "Unknown")
    do_borough = _zone_borough.get(dropoff_zone, "Unknown")

    row = np.zeros(len(get_feature_names()), dtype=np.float32)
    idx = 0

    # Baseline
    row[idx] = pickup_zone;          idx += 1
    row[idx] = dropoff_zone;         idx += 1
    row[idx] = hour;                 idx += 1
    row[idx] = dow;                  idx += 1
    row[idx] = month;                idx += 1
    row[idx] = passenger_count;      idx += 1

    # Geometry
    row[idx] = haversine;            idx += 1
    row[idx] = bearing;              idx += 1
    row[idx] = log_hav;              idx += 1

    # Temporal
    row[idx] = hour_sin;             idx += 1
    row[idx] = hour_cos;             idx += 1
    row[idx] = is_weekend;           idx += 1
    row[idx] = is_rush;              idx += 1

    # Interaction features (distance × time)
    row[idx] = haversine_x_rush;      idx += 1
    row[idx] = log_haversine_x_rush;  idx += 1
    row[idx] = haversine_x_hour_sin;  idx += 1
    row[idx] = haversine_x_hour_cos;  idx += 1

    # Borough features
    row[idx] = 1.0 if pu_borough == do_borough else 0.0;  idx += 1
    for b in sorted(_borough_to_idx):
        row[idx] = 1.0 if pu_borough == b else 0.0;  idx += 1
    for b in sorted(_borough_to_idx):
        row[idx] = 1.0 if do_borough == b else 0.0;  idx += 1

    return row.reshape(1, -1)