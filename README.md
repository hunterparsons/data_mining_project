# data_mining_project

Predicts the 2026 NCAA Men's Basketball Tournament using four models trained on historical regular-season and tournament data from Kaggle's *March Machine Learning Mania 2026* competition. Each trained model is then run through a Monte Carlo bracket simulation to estimate per-team championship odds.

## Methods

- Logistic Regression
- Random Forest
- XGBoost
- 1D CNN + Multi-Head Attention (Keras)

## Requirements

- Python 3.11.3
- Dependencies in `requirements.txt` (`pip install -r requirements.txt`)
- A Kaggle account with API credentials configured for `kagglehub`. Easiest setup:
  1. Go to your Kaggle account settings and create a new API token (downloads `kaggle.json`).
  2. Place it at `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`.
  3. Accept the competition rules at https://www.kaggle.com/competitions/march-machine-learning-mania-2026 — `kagglehub` cannot download the data otherwise.

## Running

```bash
cd src
python main.py
```

On first run, `kagglehub` downloads the competition data into its local cache. The script then trains all four models, prints validation metrics (LogLoss / Accuracy), and runs a 100-simulation Monte Carlo bracket per model, printing each team's championship probability.

## Project structure

```
src/
  main.py                          # entrypoint: trains models, runs simulations
  data/
    download.py                    # pulls the competition data via kagglehub
    traditional_aggregate.py       # season-level team stats
    traditional_processing.py      # feature engineering for LR / RF / XGB
    deep_model_processing.py       # game-sequence tensors for the deep model
    calculate_conference_strength.py
    data_2026.py                   # 2026 inference inputs
    team_mapping.py                # TeamID -> team name
  models/
    lr.py, rf.py, xgb.py, deep_model.py
  engine/
    bracket_engine.py              # bracket simulation + Monte Carlo
```

## Output

`results.txt` is a saved log from a previous full run, kept in the repo for reference. Re-running `main.py` regenerates the same kind of output to stdout.
