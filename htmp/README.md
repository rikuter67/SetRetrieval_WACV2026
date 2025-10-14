# Hull Tactical Market Prediction (HTMP) Kaggle Template

This repository provides a lightweight but fully functional Kaggle competition template tailored for the [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) challenge. It is inspired by [unonao/kaggle-template](https://github.com/unonao/kaggle-template) and is intended to be a starting point for seminars or group projects. Students can clone this template, hook up the Kaggle dataset, and start experimenting right away.

## Repository layout

```
htmp/
├── configs/                 # Experiment configuration files (YAML)
├── data/
│   ├── raw/                 # Downloaded Kaggle CSVs go here (train/test/sample_submission)
│   ├── processed/           # Intermediate feature matrices
│   └── notebooks/           # Optional exploratory datasets or exports
├── logs/                    # Training logs, CV scores, metadata
├── models/                  # Serialized model artefacts
├── notebooks/               # Jupyter notebooks used for exploration
├── scripts/                 # CLI entry points for training/inference
├── src/                     # Reusable Python modules (config, features, models)
└── submissions/             # Generated Kaggle submission files
```

## Quick start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download Kaggle data**

   Use the Kaggle CLI to download and unzip the competition data into `data/raw/`:

   ```bash
   kaggle competitions download -c hull-tactical-market-prediction -p data/raw
   unzip data/raw/hull-tactical-market-prediction.zip -d data/raw/
   ```

   Ensure the following files exist:

   ```
   data/raw/train.csv
   data/raw/test.csv
   data/raw/sample_submission.csv
   ```

3. **Run the baseline pipeline**

   The baseline uses a simple feature extractor that standardises numeric columns, fills missing values, and trains a ridge regression model with time-aware cross-validation. The commands below will train the model, cache the processed features, and create a submission file.

   ```bash
   # Train a model (with cross-validation + OOF predictions)
   python scripts/train.py --config configs/default.yaml

   # Generate predictions for the public test set
   python scripts/predict.py --config configs/default.yaml
   ```

   After running, you will find:

   - `models/default_fold_*.pkl` – per-fold trained estimators.
   - `models/default_metadata.json` – experiment metadata (CV scores, feature list, timestamps).
   - `submissions/default_submission.csv` – ready-to-upload Kaggle submission.

4. **Play with the evaluation API baseline**

   For streaming evaluation (the official Kaggle inference harness), we provide a lazy-fit
   baseline in `scripts/online_server.py`. It reads `train.csv` the first time a prediction is
   requested, trains a ridge regression on numeric columns, and then serves per-timestep
   forecasts via Kaggle's `DefaultInferenceServer`:

   ```bash
   python scripts/online_server.py
   ```

   When executed locally this launches the `run_local_gateway` helper so you can sanity check
   the server behaviour before packaging it for submission.

## Submission notebook

- `notebooks/htmp_lazy_fit_baseline.ipynb` mirrors the evaluation server logic and is ready to be
  uploaded as a Kaggle Notebook submission. It bundles the lazy-fit baseline, optional
  dependency installs, and inline commentary (in Japanese) so seminar participants can study and
  extend the approach directly inside Kaggle.

## Customisation tips

- Adjust the `configs/default.yaml` file to modify model type, feature options, or cross-validation strategy.
- Add new feature builders by extending `src/features.py` and registering them in the configuration.
- Replace the baseline ridge regression with gradient boosting (e.g. LightGBM) by tweaking `model.type` in the config and optionally tuning hyperparameters.
- Track experiments by committing the generated `logs/` metadata or uploading to your preferred experiment tracker.

## Reproducibility

- All scripts accept a `--seed` argument (default 42) to ensure reproducibility of data splits and randomised models.
- Configuration-driven workflows make it easy to version and review changes.

## License

This template is provided as-is for educational purposes within the Data Analysis 2025 seminar.
