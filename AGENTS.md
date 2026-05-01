# AI Agents & Project Instructions

## AI Tooling Used
This project was developed iteratively using a combination of powerful agentic AI models:
- **DeepSeek-V4-Pro** (via Hugging Face API in Cline)
- **Claude Opus 4.6** (in Google Antigravity)
- **Gemini 3.1 Pro** (in Google Antigravity)

These tools were pivotal for rapidly prototyping architectures (XGBoost vs. LightGBM vs. PyTorch), setting up `DataLoader` boilerplate, and ensuring the final Docker build stayed under the 2.5 GB limit by isolating a CPU-only PyTorch wheel.

## Project Context
NYC taxi trip duration prediction for the GobbleCube AI Builder hiring challenge.
**Objective**: Beat the baseline XGBoost MAE of ~350s on the held-out 2024 Dev set using richer spatial features and a PyTorch Neural Network.

## Style Guide
- Python 3.10+, use type hints on all public functions.
- 4-space indentation, 100-char line limit.
- NumPy-style docstrings for public functions and classes.
- Use `pathlib` for file paths, not `os.path`.
- Prefer vectorized `pandas`/`numpy` or PyTorch tensors over Python loops.
- No bare excepts — catch specific exceptions.

## Commands
- Run contract tests: `pytest tests/ -v`
- Run local grading: `python grade.py`
- Download data: `python data/download_data.py`
- Build Docker: `docker build -t gobblecube-eta .`

## Architecture (Root-Level)
- `predict.py` — Inference entrypoint (loaded by grader). Exposes `predict(request: dict) -> float`.
- `features.py` — Shared feature engineering module (static zone metadata, haversine, bearings).
- `train.py` — Final training script. Loads data, builds features, trains the MLP, and saves `model.pkl`.
- `grade.py` — NEVER MODIFIED. Official local scoring harness.
- `tests/test_submission.py` — Contract tests from starter. Must always pass.
- `data/download_data.py` — Downloads NYC taxi parquet files (gitignored).

## Behavioral Constraints
- `grade.py` must never be modified — it's the official scoring contract.
- `tests/test_submission.py` must always pass before any commit.
- No parquet files in git — they are explicitly excluded in `.gitignore`.
- Pickled model weights (`model.pkl`) are tracked natively by git.
- Pickle model must execute using CPU-only inference; no GPU required at runtime.
- Inference must complete in < 200ms per request.
- Docker image must be < 2.5 GB.
- No external API calls inside `predict()`.

## Data
- NYC TLC Green + Yellow Taxi, 2022–2024.
- `data/train.parquet` — Training data (downloaded).
- `data/dev.parquet` — Held-out 2024 Dev set (for local score tracking).
- `data/zone_lookup.csv` — NYC taxi zone metadata.
- `data/zone_centroids.csv` — Pre-computed zone lat/lon centroids for spatial features.