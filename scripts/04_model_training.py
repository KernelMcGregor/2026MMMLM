#!/usr/bin/env python3
"""
Phase 4: Model Training
=======================
Train SEPARATE XGBoost models for Men and Women with LOSO CV.

Inputs:
- processed/training_features.csv

Outputs:
- models/xgb_loso_models.pkl (contains separate M and W models)
- processed/oof_predictions.csv
- outputs/04_model_training.log
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = PROJECT_DIR / "models"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Logging setup
LOG_FILE = OUTPUT_DIR / "04_model_training.log"

# XGBoost parameters (from winning solutions)
XGB_PARAMS = {
    'objective': 'reg:squarederror',  # Predict point differential
    'booster': 'gbtree',
    'eta': 0.01,
    'subsample': 0.6,
    'colsample_bynode': 0.8,
    'num_parallel_tree': 2,
    'min_child_weight': 4,
    'max_depth': 4,
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
    'max_bin': 32,
    'seed': 42,
    'verbosity': 0
}

NUM_BOOST_ROUNDS = 700
EARLY_STOPPING_ROUNDS = 50


def log(message, also_print=True):
    """Log message to file and optionally print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"

    with open(LOG_FILE, "a") as f:
        f.write(log_message + "\n")

    if also_print:
        print(log_message)


def print_separator(title=""):
    """Print a visual separator."""
    sep = "=" * 70
    if title:
        log(f"\n{sep}\n{title.center(70)}\n{sep}")
    else:
        log(sep)


def train_loso_models(X, y, feature_cols, params, num_rounds, gender_label=""):
    """
    Train XGBoost with Leave-One-Season-Out cross-validation.
    """
    try:
        import xgboost as xgb
    except ImportError:
        log("ERROR: xgboost not installed. Install with: pip install xgboost")
        sys.exit(1)

    seasons = sorted(X['Season'].unique())
    models = {}
    oof_predictions = []
    oof_targets = []
    oof_seasons = []
    oof_teams = []

    gender_str = f" ({gender_label})" if gender_label else ""
    log(f"\nTraining{gender_str} with LOSO CV across {len(seasons)} seasons...")
    log(f"Features: {len(feature_cols)}")
    log(f"Samples: {len(X):,}")
    log(f"Early stopping rounds: {EARLY_STOPPING_ROUNDS}")

    for holdout_season in seasons:
        log(f"\n--- Holdout Season: {holdout_season} ---")

        train_mask = X['Season'] != holdout_season
        val_mask = X['Season'] == holdout_season

        X_train = X.loc[train_mask, feature_cols].values
        y_train = y[train_mask].values
        X_val = X.loc[val_mask, feature_cols].values
        y_val = y[val_mask].values

        if len(X_val) == 0:
            log(f"  No validation data for season {holdout_season}, skipping")
            continue

        log(f"  Training samples: {len(X_train):,}")
        log(f"  Validation samples: {len(X_val):,}")

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        # Train model
        evals_result = {}
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals_result=evals_result,
            verbose_eval=False
        )

        models[holdout_season] = model

        # Get predictions
        preds = model.predict(dval)
        oof_predictions.extend(preds)
        oof_targets.extend(y_val)
        oof_seasons.extend([holdout_season] * len(preds))

        # Get team IDs for this validation set
        val_t1 = X.loc[val_mask, 'T1_TeamID'].values if 'T1_TeamID' in X.columns else [0] * len(preds)
        val_t2 = X.loc[val_mask, 'T2_TeamID'].values if 'T2_TeamID' in X.columns else [0] * len(preds)
        oof_teams.extend(list(zip(val_t1, val_t2)))

        # Calculate metrics
        mae = np.mean(np.abs(preds - y_val))
        rmse = np.sqrt(np.mean((preds - y_val) ** 2))
        best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else num_rounds

        log(f"  Best iteration: {best_iteration}")
        log(f"  Validation MAE: {mae:.3f}")
        log(f"  Validation RMSE: {rmse:.3f}")

        # Feature importance for this fold
        importance = model.get_score(importance_type='gain')
        if importance:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            log(f"  Top 5 features (gain):")
            for feat, score in top_features:
                log(f"    - {feat}: {score:.1f}")

    return models, np.array(oof_predictions), np.array(oof_targets), np.array(oof_seasons), oof_teams


def compute_metrics(predictions, targets, seasons=None):
    """Compute various evaluation metrics."""
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # Correlation
    corr = np.corrcoef(predictions, targets)[0, 1]

    # Win prediction accuracy (predict T1 wins if predicted margin > 0)
    pred_wins = (predictions > 0).astype(int)
    actual_wins = (targets > 0).astype(int)
    accuracy = np.mean(pred_wins == actual_wins)

    # Brier score (using sigmoid to convert margin to probability)
    def margin_to_prob(margin, scale=10):
        return 1 / (1 + np.exp(-margin / scale))

    pred_probs = margin_to_prob(predictions)
    brier = np.mean((pred_probs - actual_wins) ** 2)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Correlation': corr,
        'Accuracy': accuracy,
        'Brier': brier
    }


def main():
    """Main execution function."""
    # Clear previous log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    print_separator("PHASE 4: MODEL TRAINING")

    # =========================================================================
    # LOAD TRAINING DATA
    # =========================================================================
    print_separator("Loading Training Data")

    training_path = PROCESSED_DIR / "training_features.csv"

    if not training_path.exists():
        log(f"ERROR: {training_path} not found. Run 03_feature_engineering.py first.")
        sys.exit(1)

    training_df = pd.read_csv(training_path)
    log(f"Loaded training features: {len(training_df):,} samples")

    # Identify feature columns (exclude Gender - we train separate models)
    exclude_cols = ['Season', 'T1_TeamID', 'T2_TeamID', 'PointDiff', 'T1_Win', 'Gender']
    feature_cols = [c for c in training_df.columns if c not in exclude_cols]

    log(f"Feature columns: {len(feature_cols)}")
    log(f"Seasons in data: {sorted(training_df['Season'].unique())}")

    # Check for Gender column
    if 'Gender' in training_df.columns:
        genders = sorted(training_df['Gender'].unique())
        log(f"Genders in data: {genders}")
        log("NOTE: Training SEPARATE models for each gender")
    else:
        # Infer gender from team IDs
        training_df['Gender'] = training_df['T1_TeamID'].apply(lambda x: 'M' if x < 2000 else 'W')
        genders = sorted(training_df['Gender'].unique())
        log(f"Inferred genders from TeamID: {genders}")

    # Handle missing values
    missing_before = training_df[feature_cols].isna().sum().sum()
    log(f"Missing values before imputation: {missing_before}")

    # Fill missing values with column means
    for col in feature_cols:
        if training_df[col].isna().any():
            fill_value = training_df[col].mean()
            training_df[col] = training_df[col].fillna(fill_value)
            log(f"  Filled {col} missing values with mean: {fill_value:.3f}")

    # Prepare X and y
    X = training_df.copy()
    y = training_df['PointDiff']

    log(f"\nTarget (PointDiff) statistics:")
    log(f"  Mean: {y.mean():.2f}")
    log(f"  Std: {y.std():.2f}")
    log(f"  Min: {y.min():.0f}")
    log(f"  Max: {y.max():.0f}")

    # =========================================================================
    # TRAIN MODELS (SEPARATE FOR EACH GENDER)
    # =========================================================================
    all_models = {}  # {(gender, season): model}
    all_oof_predictions = []
    all_oof_targets = []
    all_oof_seasons = []
    all_oof_genders = []
    all_oof_teams = []

    for gender in genders:
        print_separator(f"Training XGBoost Models - {gender} (LOSO CV)")

        # Filter to this gender
        gender_mask = training_df['Gender'] == gender
        X_gender = X[gender_mask].copy()
        y_gender = y[gender_mask]

        log(f"\n{gender} dataset: {len(X_gender):,} samples")
        log(f"Seasons: {sorted(X_gender['Season'].unique())}")

        models, oof_predictions, oof_targets, oof_seasons, oof_teams = train_loso_models(
            X_gender, y_gender, feature_cols, XGB_PARAMS, NUM_BOOST_ROUNDS, gender_label=gender
        )

        # Store models with gender key
        for season, model in models.items():
            all_models[(gender, season)] = model

        # Store OOF predictions with gender
        all_oof_predictions.extend(oof_predictions)
        all_oof_targets.extend(oof_targets)
        all_oof_seasons.extend(oof_seasons)
        all_oof_genders.extend([gender] * len(oof_predictions))
        all_oof_teams.extend(oof_teams)

        log(f"\nTrained {len(models)} {gender} models")

    # Convert to arrays
    oof_predictions = np.array(all_oof_predictions)
    oof_targets = np.array(all_oof_targets)
    oof_seasons = np.array(all_oof_seasons)
    oof_genders = np.array(all_oof_genders)
    oof_teams = all_oof_teams

    log(f"\n\nTotal models trained: {len(all_models)}")

    # =========================================================================
    # EVALUATE OOF PREDICTIONS
    # =========================================================================
    print_separator("Out-of-Fold Evaluation")

    overall_metrics = compute_metrics(oof_predictions, oof_targets)

    log("\nOverall OOF Metrics:")
    for metric, value in overall_metrics.items():
        log(f"  {metric}: {value:.4f}")

    # Per-gender metrics
    log("\nPer-Gender Metrics:")
    for gender in genders:
        mask = oof_genders == gender
        if mask.sum() > 0:
            metrics = compute_metrics(oof_predictions[mask], oof_targets[mask])
            log(f"  {gender}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, "
                f"Acc={metrics['Accuracy']*100:.1f}%, Brier={metrics['Brier']:.4f}, N={mask.sum()}")

    # Per-season metrics (combined)
    log("\nPer-Season Metrics:")
    unique_seasons = sorted(set(oof_seasons))

    season_metrics = []
    for season in unique_seasons:
        mask = oof_seasons == season
        if mask.sum() > 0:
            metrics = compute_metrics(oof_predictions[mask], oof_targets[mask])
            metrics['Season'] = season
            metrics['N'] = mask.sum()
            season_metrics.append(metrics)

            log(f"  Season {season}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, "
                f"Acc={metrics['Accuracy']*100:.1f}%, Brier={metrics['Brier']:.4f}, N={metrics['N']}")

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    print_separator("Feature Importance Analysis")

    # Aggregate importance across all models
    all_importance = {}

    for (gender, season), model in all_models.items():
        importance = model.get_score(importance_type='gain')
        for feat, score in importance.items():
            if feat not in all_importance:
                all_importance[feat] = []
            all_importance[feat].append(score)

    # Average importance
    avg_importance = {feat: np.mean(scores) for feat, scores in all_importance.items()}
    sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    log("\nTop 20 Features by Average Gain:")
    for i, (feat, score) in enumerate(sorted_importance[:20], 1):
        log(f"  {i:2d}. {feat}: {score:.2f}")

    # Features never used
    unused_features = [f for f in feature_cols if f not in all_importance]
    if unused_features:
        log(f"\nFeatures never used by any model: {unused_features}")

    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================
    print_separator("Error Analysis")

    # Distribution of errors
    errors = oof_predictions - oof_targets
    log(f"\nPrediction Error Distribution:")
    log(f"  Mean error: {errors.mean():.3f}")
    log(f"  Std error: {errors.std():.3f}")
    log(f"  25th percentile: {np.percentile(errors, 25):.3f}")
    log(f"  50th percentile: {np.percentile(errors, 50):.3f}")
    log(f"  75th percentile: {np.percentile(errors, 75):.3f}")

    # Largest errors
    abs_errors = np.abs(errors)
    log(f"\nLargest Absolute Errors:")
    top_error_indices = np.argsort(abs_errors)[-10:][::-1]
    for idx in top_error_indices:
        log(f"  Season {oof_seasons[idx]}: Predicted {oof_predictions[idx]:.1f}, "
            f"Actual {oof_targets[idx]:.1f}, Error {errors[idx]:.1f}")

    # Upset analysis
    upset_mask = (oof_predictions > 0) != (oof_targets > 0)
    num_upsets = upset_mask.sum()
    log(f"\nIncorrect Win Predictions (upsets): {num_upsets} / {len(oof_predictions)} "
        f"({num_upsets/len(oof_predictions)*100:.1f}%)")

    # =========================================================================
    # SAVE MODELS AND PREDICTIONS
    # =========================================================================
    print_separator("Saving Models and Predictions")

    # Save models (keyed by (gender, season))
    model_data = {
        'models': all_models,  # {(gender, season): model}
        'feature_cols': feature_cols,
        'xgb_params': XGB_PARAMS,
        'num_boost_rounds': NUM_BOOST_ROUNDS,
        'genders': genders
    }

    with open(MODELS_DIR / "xgb_loso_models.pkl", "wb") as f:
        pickle.dump(model_data, f)
    log(f"Saved: {MODELS_DIR / 'xgb_loso_models.pkl'}")
    log(f"  Models: {len(all_models)} ({len(genders)} genders x {len(all_models)//len(genders)} seasons)")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        'Season': oof_seasons,
        'Gender': oof_genders,
        'Prediction': oof_predictions,
        'Actual': oof_targets,
        'Error': errors,
        'AbsError': abs_errors
    })

    oof_df.to_csv(PROCESSED_DIR / "oof_predictions.csv", index=False)
    log(f"Saved: {PROCESSED_DIR / 'oof_predictions.csv'}")

    # Save feature importance
    importance_df = pd.DataFrame([
        {'Feature': feat, 'AvgGain': score, 'NumModels': len(all_importance.get(feat, []))}
        for feat, score in sorted_importance
    ])
    importance_df.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)
    log(f"Saved: {PROCESSED_DIR / 'feature_importance.csv'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_separator("PHASE 4 COMPLETE - SUMMARY")

    log(f"\nModel Training Summary:")
    log(f"  - Genders: {genders}")
    log(f"  - Total models trained: {len(all_models)}")
    for gender in genders:
        gender_models = [k for k in all_models.keys() if k[0] == gender]
        log(f"    - {gender} models: {len(gender_models)}")
    log(f"  - Features used: {len(feature_cols)}")
    log(f"  - Training samples: {len(training_df):,}")
    log(f"  - OOF samples: {len(oof_predictions):,}")

    log(f"\nKey Metrics:")
    log(f"  - OOF MAE: {overall_metrics['MAE']:.3f} points")
    log(f"  - OOF RMSE: {overall_metrics['RMSE']:.3f} points")
    log(f"  - Win Prediction Accuracy: {overall_metrics['Accuracy']*100:.1f}%")
    log(f"  - Brier Score: {overall_metrics['Brier']:.4f}")

    log(f"\nTop 5 Most Important Features:")
    for i, (feat, score) in enumerate(sorted_importance[:5], 1):
        log(f"  {i}. {feat}")

    log(f"\nOutputs saved to: {MODELS_DIR} and {PROCESSED_DIR}")
    log(f"Log saved to: {LOG_FILE}")

    print_separator()

    return 0


if __name__ == "__main__":
    sys.exit(main())
