# Cline Instructions

## Project Context
NYC taxi trip duration prediction for GobbleCube AI Builder hiring challenge.
Beat the baseline XGBoost MAE of ~350s on held-out 2024 Dev set using richer features and LightGBM.

## Style Guide
- Python 3.10+, use type hints on all public functions
- 4-space indentation, 100-char line limit
- NumPy-style docstrings for public functions
- Use pathlib for file paths, not os.path
- Prefer vectorized pandas/NumPy over Python loops
- No bare excepts — catch specific exceptions

## Commands
- Run contract tests: `pytest tests/ -v`
- Run local grading: `python src/grade.py`
- Download data: `python data/download_data.py`
- Build Docker: `docker build -t gobblecube-eta .`

## Architecture
- `src/predict.py` — Inference entrypoint (loaded by grader). Must expose `predict(request: dict) -> float`
- `src/features.py` — Shared feature engineering module. Imported by experiments, train.py, and predict.py
- `src/train.py` — Final training script. Loads data, builds features, saves model.pkl
- `src/grade.py` — NEVER MODIFIED. Official local scoring harness
- `tests/test_submission.py` — Contract tests from starter. Must always pass
- `data/download_data.py` — Downloads NYC taxi parquet files (gitignored)
- `_work/experiments/` — Gitignored. All experimental scripts live here

## Behavioral Constraints
- `src/grade.py` must never be modified — it's the official scoring contract
- `tests/test_submission.py` must always pass before any commit
- No parquet files in git — they are in .gitignore
- `models/model.pkl` is gitignored during development, force-added at final submission
- All experiments go in gitignored `_work/` folder, never committed
- Only winning approach gets promoted to `src/train.py` and `src/predict.py`
- Pickle model must be CPU-only inference, no GPU required at runtime
- Inference must complete in < 200ms per request
- No external API calls in predict()

## Data
- NYC TLC Green + Yellow Taxi, 2022–2024
- `data/train.parquet` — Training data (download with download_data.py)
- `data/dev.parquet` — Held-out 2024 Dev set (for score tracking)
- `data/sample_1M.parquet` — 1M row sample for fast experimentation
- `data/zone_lookup.csv` — NYC taxi zone metadata (to be downloaded)
- `data/zone_centroids.csv` — Pre-computed zone lat/lon centroids (to be computed)