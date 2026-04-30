# GobbleCube ETA Challenge

NYC taxi trip duration prediction — GobbleCube AI Builder hiring challenge.

**Goal:** Predict trip duration (seconds) from pickup zone, dropoff zone, timestamp, and passenger count. Beat the baseline MAE of ~350s on the held-out 2024 Dev set.

## Approach

Experiment-driven tabular ML with progressively richer features:

| Experiment | Description | Dev MAE (s) |
|-----------|-------------|-------------|
| exp01 | Baseline (XGBoost, 6 features) | TBD |
| exp02 | Zone geometry features (haversine, bearing, boroughs) | TBD |
| exp03 | Temporal features (sin/cos cyclic encoding) | TBD |
| exp04 | Zone-pair aggregate statistics | TBD |
| exp05 | LightGBM + Optuna hyperparameter tuning | TBD |
| exp06 | XGBoost + Optuna tuning (optional) | TBD |
| exp07 | Neural zone embeddings (optional) | TBD |

## Setup

See [SETUP.md](SETUP.md) for reproduction instructions.

## Constraints

- ≤ 200ms per inference request
- ≤ 2.5 GB Docker image
- ≤ 10 min Docker build
- No external API calls at inference
- No training on Eval set

## Challenge Context

Full challenge details: [arena/README.md](https://github.com/varsan-segar/gobblecube-eta-challenge)

Data source: NYC TLC Trip Record Data (Green + Yellow Taxi), 2022-2024.

## AI Tooling

Developed with Cline using Hugging Face DeepSeek-V4-Pro. See [AGENTS.md](AGENTS.md) for details.