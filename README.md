# GobbleCube ETA Challenge Submission

This repository contains my submission for the GobbleCube ETA Challenge.

---

## Your final score

Dev MAE: **265.6 s**

---

## Your approach, in one paragraph

I built a lightweight deep neural network using PyTorch. Instead of relying on tree-based models which struggle to interpret categorical geographic IDs numerically, the model learns a 64-dimensional embedding for each NYC taxi zone. The inputs (pickup/dropoff zone embeddings, continuous temporal/geographic features, and one-hot borough flags) are concatenated and passed through a 3-layer MLP ([256, 128, 64] with BatchNorm and ReLU). The model is extremely efficient (only ~116k parameters) and was trained with a simple L1 Loss to directly optimize the MAE metric. 

## What you tried that didn't work

1. **"Enhanced" Neural Architectures**: I tried making the NN deeper (4 layers) and added Residual blocks, SiLU activations, and LayerNorm. This massively overfit the training data (dropping train loss significantly) but failed to generalize, pushing the Validation MAE up by 3-4 seconds.
2. **LightGBM / XGBoost Feature Engineering**: Tree models (even heavily tuned via Optuna) bottomed out at around ~289s. Without learning spatial embeddings, the tree models couldn't interpolate the geographical relationships effectively, no matter how many interactions (e.g. `haversine_km * is_rush_hour`) I added.

## Where AI tooling sped you up most

I used a combination of **DeepSeek-V4-Pro** (via Hugging Face API in Cline), **Claude Opus 4.6** (in Google Antigravity), and **Gemini 3.1 Pro** (in Google Antigravity). The AI sped up my workflow significantly when experimenting with different algorithms and modeling approaches—allowing me to rapidly pivot between XGBoost, LightGBM, and various Neural Network architectures to test what worked best. It was also exceptionally fast at setting up the PyTorch `Dataset` / `DataLoader` boilerplate and constructing the training loop. Finally, it shined during the Dockerization phase—instantly recognizing that `pip install torch` would download massive GPU binaries exceeding the 2.5 GB container limit, and swapping it to the `--index-url https://download.pytorch.org/whl/cpu` wheel to keep the final image perfectly lean.

## Next experiments

If I kept going, I would compute historical zone-pair average velocities. The network currently learns static embeddings, but taxi speeds are highly dynamic. Adding a feature for "average trip duration from zone A to B during hour H over the last 30 days" would likely provide a massive signal boost and bridge the remaining gap to the theoretical minimum error.

## How to reproduce

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Note: To run train.py, you will also need to install PyTorch manually:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Download data
python data/download_data.py

# 3. Train Neural Network (outputs model.pkl)
python train.py
```

---

*Total time spent on this challenge: ~1 day.*