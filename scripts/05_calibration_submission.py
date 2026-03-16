#!/usr/bin/env python3
"""
Phase 5: Calibration & Submission (OPTIMIZED + PROB PUSHING)
=============================================================
Fit advanced calibration, apply probability pushing, generate predictions.

Features:
- Multiple calibration methods (logistic, isotonic, beta)
- Probability pushing to maximize confident predictions
- Seed-based adjustments for historical upset rates

Inputs:
- models/xgb_loso_models.pkl
- processed/oof_predictions.csv
- processed/season_stats.csv
- processed/all_ratings.csv
- processed/gold_medal_features.pkl
- processed/seeds.csv

Outputs:
- models/calibration.pkl
- submissions/submission.csv
- submissions/submission_pushed.csv (aggressive version)
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = PROJECT_DIR / "models"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSIONS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Logging setup
LOG_FILE = OUTPUT_DIR / "05_calibration_submission.log"


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


# =============================================================================
# CALIBRATION METHODS
# =============================================================================

def fit_logistic_calibration(predictions, targets):
    """Fit logistic calibration with optimal scale and bias."""
    binary_targets = (targets > 0).astype(float)

    def brier_loss(params):
        scale, bias = params
        probs = 1 / (1 + np.exp(-(predictions + bias) / scale))
        return np.mean((probs - binary_targets) ** 2)

    # Optimize scale and bias
    result = minimize(brier_loss, [10.0, 0.0], method='Nelder-Mead')
    best_scale, best_bias = result.x
    best_brier = result.fun

    log(f"  Logistic: scale={best_scale:.2f}, bias={best_bias:.2f}, Brier={best_brier:.4f}")

    return {'type': 'logistic', 'scale': best_scale, 'bias': best_bias, 'brier': best_brier}


def fit_isotonic_calibration(predictions, targets):
    """Fit isotonic regression for non-parametric calibration."""
    binary_targets = (targets > 0).astype(float)

    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(predictions, binary_targets)

    # Evaluate
    probs = iso_reg.predict(predictions)
    brier = np.mean((probs - binary_targets) ** 2)

    log(f"  Isotonic: Brier={brier:.4f}")

    return {'type': 'isotonic', 'model': iso_reg, 'brier': brier}


def fit_beta_calibration(predictions, targets):
    """Fit beta calibration (3-parameter) for better extreme probabilities."""
    binary_targets = (targets > 0).astype(float)

    # First convert margins to rough probabilities using logistic
    rough_probs = 1 / (1 + np.exp(-predictions / 10))

    # Beta calibration: p_cal = 1 / (1 + (1/p - 1)^c * exp(a + b*p))
    # Simplified: use power transform on logit scale
    def beta_loss(params):
        a, b, c = params
        # Transform: apply power to odds ratio
        eps = 1e-6
        rough_clipped = np.clip(rough_probs, eps, 1-eps)
        logit = np.log(rough_clipped / (1 - rough_clipped))
        transformed_logit = a + b * logit + c * np.sign(logit) * np.abs(logit) ** 1.5
        probs = 1 / (1 + np.exp(-transformed_logit))
        probs = np.clip(probs, eps, 1-eps)
        return np.mean((probs - binary_targets) ** 2)

    result = minimize(beta_loss, [0.0, 1.0, 0.0], method='Nelder-Mead')
    a, b, c = result.x
    brier = result.fun

    log(f"  Beta: a={a:.3f}, b={b:.3f}, c={c:.3f}, Brier={brier:.4f}")

    return {'type': 'beta', 'a': a, 'b': b, 'c': c, 'brier': brier}


def fit_all_calibrations(predictions, targets):
    """Fit all calibration methods and return the best one."""
    log("\nFitting multiple calibration methods...")

    logistic = fit_logistic_calibration(predictions, targets)
    isotonic = fit_isotonic_calibration(predictions, targets)
    beta = fit_beta_calibration(predictions, targets)

    # Pick best by Brier score
    methods = [logistic, isotonic, beta]
    best = min(methods, key=lambda x: x['brier'])

    log(f"\n  Best method: {best['type']} (Brier={best['brier']:.4f})")

    return best, {'logistic': logistic, 'isotonic': isotonic, 'beta': beta}


def apply_calibration(margins, calibration_model):
    """Convert margins to probabilities using fitted calibration."""
    cal_type = calibration_model['type']

    if cal_type == 'logistic':
        scale = calibration_model['scale']
        bias = calibration_model.get('bias', 0)
        probs = 1 / (1 + np.exp(-(margins + bias) / scale))

    elif cal_type == 'isotonic':
        probs = calibration_model['model'].predict(margins)

    elif cal_type == 'beta':
        a, b, c = calibration_model['a'], calibration_model['b'], calibration_model['c']
        rough_probs = 1 / (1 + np.exp(-margins / 10))
        eps = 1e-6
        rough_clipped = np.clip(rough_probs, eps, 1-eps)
        logit = np.log(rough_clipped / (1 - rough_clipped))
        transformed_logit = a + b * logit + c * np.sign(logit) * np.abs(logit) ** 1.5
        probs = 1 / (1 + np.exp(-transformed_logit))

    return np.clip(probs, 0.01, 0.99)


# =============================================================================
# PROBABILITY PUSHING
# =============================================================================

def push_probabilities(probs, strength=1.5, method='power'):
    """
    Push confident predictions toward extremes.

    Methods:
    - 'power': Apply power transform to push away from 0.5
    - 'sigmoid': Apply sigmoid stretch to amplify confidence
    - 'linear': Linear stretch toward extremes

    strength: How aggressively to push (1.0 = no change, 2.0 = aggressive)
    """
    centered = probs - 0.5  # Center around 0

    if method == 'power':
        # Power transform: sign(x) * |x|^(1/strength)
        # strength > 1 pushes toward extremes
        pushed = np.sign(centered) * np.abs(centered * 2) ** (1 / strength) / 2

    elif method == 'sigmoid':
        # Sigmoid stretch: apply logistic transform with steeper slope
        logit = np.log((probs + 1e-6) / (1 - probs + 1e-6))
        stretched_logit = logit * strength
        pushed = 1 / (1 + np.exp(-stretched_logit)) - 0.5

    elif method == 'linear':
        # Simple linear stretch
        pushed = centered * strength

    result = pushed + 0.5
    return np.clip(result, 0.01, 0.99)


def apply_seed_adjustments(probs, t1_seeds, t2_seeds):
    """
    Apply historical seed-based upset probability adjustments.
    Based on historical NCAA tournament upset rates.
    """
    # Historical upset rates by seed matchup (approximate)
    # Format: (higher_seed, lower_seed) -> historical win rate for higher seed
    HISTORICAL_UPSETS = {
        (1, 16): 0.99,  # 1 seeds almost never lose
        (2, 15): 0.94,
        (3, 14): 0.85,
        (4, 13): 0.79,
        (5, 12): 0.65,  # 12 vs 5 is famous upset territory
        (6, 11): 0.63,
        (7, 10): 0.61,
        (8, 9): 0.51,   # Basically a coin flip
    }

    adjusted = probs.copy()

    for i in range(len(probs)):
        s1, s2 = int(t1_seeds[i]), int(t2_seeds[i])
        if s1 == s2:
            continue

        # Determine which is favored by seed
        if s1 < s2:  # T1 is higher seed (lower number = better)
            matchup = (s1, s2)
            if matchup in HISTORICAL_UPSETS:
                hist_rate = HISTORICAL_UPSETS[matchup]
                # Blend model prediction with historical rate
                # Weight historical more when model is unsure
                model_confidence = abs(probs[i] - 0.5) * 2  # 0-1 scale
                blend_weight = 0.3 * (1 - model_confidence)  # Less blending when confident
                adjusted[i] = probs[i] * (1 - blend_weight) + hist_rate * blend_weight
        else:  # T2 is higher seed
            matchup = (s2, s1)
            if matchup in HISTORICAL_UPSETS:
                hist_rate = 1 - HISTORICAL_UPSETS[matchup]  # Flip for T1 perspective
                model_confidence = abs(probs[i] - 0.5) * 2
                blend_weight = 0.3 * (1 - model_confidence)
                adjusted[i] = probs[i] * (1 - blend_weight) + hist_rate * blend_weight

    return np.clip(adjusted, 0.01, 0.99)


def build_features_vectorized(submission_df, all_ratings, season_stats,
                               gold_medal_features, seeds, massey_ordinals=None):
    """
    Build features for all matchups using VECTORIZED pandas operations.
    Much faster than row-by-row iteration.
    """
    log("Building features using vectorized operations...")

    df = submission_df.copy()

    # Parse IDs
    parts = df['ID'].str.split('_', expand=True)
    df['Season'] = parts[0].astype(int)
    df['T1_TeamID'] = parts[1].astype(int)
    df['T2_TeamID'] = parts[2].astype(int)

    # Men/Women indicator
    df['men_women'] = (df['T1_TeamID'] < 2000).astype(int)

    # Merge ratings for T1
    ratings_t1 = all_ratings.rename(columns={
        'TeamID': 'T1_TeamID',
        'BT_strength': 'T1_bt',
        'Elo': 'T1_elo',
        'GLM_quality': 'T1_quality'
    })
    df = df.merge(ratings_t1[['Season', 'T1_TeamID', 'T1_bt', 'T1_elo', 'T1_quality']],
                  on=['Season', 'T1_TeamID'], how='left')

    # Merge ratings for T2
    ratings_t2 = all_ratings.rename(columns={
        'TeamID': 'T2_TeamID',
        'BT_strength': 'T2_bt',
        'Elo': 'T2_elo',
        'GLM_quality': 'T2_quality'
    })
    df = df.merge(ratings_t2[['Season', 'T2_TeamID', 'T2_bt', 'T2_elo', 'T2_quality']],
                  on=['Season', 'T2_TeamID'], how='left')

    # Compute diffs
    df['bt_diff'] = df['T1_bt'] - df['T2_bt']
    df['elo_diff'] = df['T1_elo'] - df['T2_elo']
    df['quality_diff'] = df['T1_quality'] - df['T2_quality']

    # Merge seeds for T1
    if len(seeds) > 0 and 'SeedNum' in seeds.columns:
        seeds_t1 = seeds[['Season', 'TeamID', 'SeedNum']].rename(columns={
            'TeamID': 'T1_TeamID',
            'SeedNum': 'T1_seed'
        })
        df = df.merge(seeds_t1, on=['Season', 'T1_TeamID'], how='left')

        seeds_t2 = seeds[['Season', 'TeamID', 'SeedNum']].rename(columns={
            'TeamID': 'T2_TeamID',
            'SeedNum': 'T2_seed'
        })
        df = df.merge(seeds_t2, on=['Season', 'T2_TeamID'], how='left')
    else:
        df['T1_seed'] = 8
        df['T2_seed'] = 8

    df['T1_seed'] = df['T1_seed'].fillna(16)
    df['T2_seed'] = df['T2_seed'].fillna(16)
    df['Seed_diff'] = df['T2_seed'] - df['T1_seed']

    # Merge season stats for T1
    stats_cols = ['avg_Score', 'avg_FGA', 'avg_Blk', 'avg_PF',
                  'avg_opponent_Score', 'avg_opponent_FGA', 'avg_opponent_Blk', 'avg_opponent_PF',
                  'avg_PointDiff', 'WinPct']

    available_stats = [c for c in stats_cols if c in season_stats.columns]

    if available_stats:
        stats_t1 = season_stats[['Season', 'TeamID'] + available_stats].copy()
        stats_t1 = stats_t1.rename(columns={'TeamID': 'T1_TeamID'})
        stats_t1 = stats_t1.rename(columns={c: f'T1_{c}' for c in available_stats})
        df = df.merge(stats_t1, on=['Season', 'T1_TeamID'], how='left')

        stats_t2 = season_stats[['Season', 'TeamID'] + available_stats].copy()
        stats_t2 = stats_t2.rename(columns={'TeamID': 'T2_TeamID'})
        stats_t2 = stats_t2.rename(columns={c: f'T2_{c}' for c in available_stats})
        df = df.merge(stats_t2, on=['Season', 'T2_TeamID'], how='left')

    # Gold-medal features (these are dicts, need to map)
    win_ratio_14d = gold_medal_features.get('win_ratio_14d', {})
    away_wins = gold_medal_features.get('away_wins', {})
    weighted_wins = gold_medal_features.get('weighted_wins', {})

    df['T1_WinRatio14d'] = df.apply(
        lambda r: win_ratio_14d.get((r['Season'], r['T1_TeamID']), 0.5), axis=1)
    df['T2_WinRatio14d'] = df.apply(
        lambda r: win_ratio_14d.get((r['Season'], r['T2_TeamID']), 0.5), axis=1)
    df['T1_away_wins'] = df.apply(
        lambda r: away_wins.get((r['Season'], r['T1_TeamID']), 0), axis=1)
    df['T2_away_wins'] = df.apply(
        lambda r: away_wins.get((r['Season'], r['T2_TeamID']), 0), axis=1)
    df['T1_weighted_wins'] = df.apply(
        lambda r: weighted_wins.get((r['Season'], r['T1_TeamID']), 0), axis=1)
    df['T2_weighted_wins'] = df.apply(
        lambda r: weighted_wins.get((r['Season'], r['T2_TeamID']), 0), axis=1)

    # Massey Ordinals (KenPom, Sagarin, RPI, etc.)
    if massey_ordinals is not None and len(massey_ordinals) > 0:
        rank_cols = [c for c in massey_ordinals.columns if c.startswith('Rank_')]

        for col in rank_cols:
            # Merge for T1
            ordinals_t1 = massey_ordinals[['Season', 'TeamID', col]].rename(columns={
                'TeamID': 'T1_TeamID',
                col: f'T1_{col}'
            })
            df = df.merge(ordinals_t1, on=['Season', 'T1_TeamID'], how='left')

            # Merge for T2
            ordinals_t2 = massey_ordinals[['Season', 'TeamID', col]].rename(columns={
                'TeamID': 'T2_TeamID',
                col: f'T2_{col}'
            })
            df = df.merge(ordinals_t2, on=['Season', 'T2_TeamID'], how='left')

            # Compute diff (positive = T1 is better/lower rank)
            df[f'{col}_diff'] = df[f'T2_{col}'] - df[f'T1_{col}']

        log(f"  Added Massey Ordinals: {rank_cols}")

    # Add Gender column (M for TeamID < 2000, W for TeamID >= 3000)
    df['Gender'] = df['T1_TeamID'].apply(lambda x: 'M' if x < 2000 else 'W')

    # Fill NaN with defaults
    df = df.fillna(0)

    log(f"  Built features for {len(df):,} matchups")
    log(f"  Men's matchups: {(df['Gender'] == 'M').sum():,}")
    log(f"  Women's matchups: {(df['Gender'] == 'W').sum():,}")

    return df


def main():
    """Main execution function."""
    # Clear previous log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    print_separator("PHASE 5: CALIBRATION & SUBMISSION")

    # =========================================================================
    # LOAD MODELS AND DATA
    # =========================================================================
    print_separator("Loading Models and Data")

    # Load trained models
    models_path = MODELS_DIR / "xgb_loso_models.pkl"
    if not models_path.exists():
        log(f"ERROR: {models_path} not found. Run 04_model_training.py first.")
        sys.exit(1)

    with open(models_path, "rb") as f:
        model_data = pickle.load(f)

    models = model_data['models']
    feature_cols = model_data['feature_cols']
    log(f"Loaded {len(models)} XGBoost models")
    log(f"Feature columns: {len(feature_cols)}")

    # Load OOF predictions for calibration
    oof_path = PROCESSED_DIR / "oof_predictions.csv"
    if oof_path.exists():
        oof_df = pd.read_csv(oof_path)
        log(f"Loaded OOF predictions: {len(oof_df):,} samples")
    else:
        log("WARNING: OOF predictions not found")
        oof_df = pd.DataFrame()

    # Load all ratings (combined CSV)
    all_ratings = pd.read_csv(PROCESSED_DIR / "all_ratings.csv")
    log(f"Loaded ratings: {len(all_ratings):,} team-seasons")

    # Load gold-medal features
    with open(PROCESSED_DIR / "gold_medal_features.pkl", "rb") as f:
        gold_medal_features = pickle.load(f)
    log(f"Loaded gold-medal features")

    # Load season stats
    season_stats = pd.read_csv(PROCESSED_DIR / "season_stats.csv")
    log(f"Loaded season stats: {len(season_stats):,} team-seasons")

    # Load seeds
    seeds_path = PROCESSED_DIR / "seeds.csv"
    if seeds_path.exists():
        seeds = pd.read_csv(seeds_path)
        log(f"Loaded seeds: {len(seeds):,} entries")
    else:
        seeds = pd.DataFrame()
        log("WARNING: Seeds not found")

    # Load Massey Ordinals
    massey_path = PROCESSED_DIR / "massey_ordinals.csv"
    if massey_path.exists():
        massey_ordinals = pd.read_csv(massey_path)
        log(f"Loaded Massey Ordinals: {len(massey_ordinals):,} team-seasons")
        rank_cols = [c for c in massey_ordinals.columns if c.startswith('Rank_')]
        log(f"  Systems: {rank_cols}")
    else:
        massey_ordinals = pd.DataFrame()
        log("WARNING: Massey Ordinals not found - run 03_feature_engineering.py")

    # Check model structure (new format uses (gender, season) keys)
    sample_key = list(models.keys())[0]
    if isinstance(sample_key, tuple):
        log(f"Models keyed by (Gender, Season) - separate M/W models")
        genders_in_models = sorted(set(k[0] for k in models.keys()))
        log(f"  Genders: {genders_in_models}")
    else:
        log(f"Models keyed by Season only - combined M/W models (legacy)")
        genders_in_models = None

    # =========================================================================
    # FIT CALIBRATION (MULTIPLE METHODS)
    # =========================================================================
    print_separator("Fitting Calibration Models")

    if len(oof_df) > 0:
        best_calibration, all_calibrations = fit_all_calibrations(
            oof_df['Prediction'].values,
            oof_df['Actual'].values
        )

        # Also test probability pushing on OOF data
        log("\nTesting probability pushing on OOF data...")
        binary_targets = (oof_df['Actual'].values > 0).astype(float)
        base_probs = apply_calibration(oof_df['Prediction'].values, best_calibration)

        push_results = []
        for strength in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
            for method in ['power', 'sigmoid']:
                pushed = push_probabilities(base_probs, strength=strength, method=method)
                brier = np.mean((pushed - binary_targets) ** 2)
                push_results.append({
                    'strength': strength,
                    'method': method,
                    'brier': brier
                })

        # Find best pushing config
        best_push = min(push_results, key=lambda x: x['brier'])
        log(f"  Best pushing: {best_push['method']} strength={best_push['strength']:.1f}, Brier={best_push['brier']:.4f}")

        # Compare to no pushing
        no_push_brier = np.mean((base_probs - binary_targets) ** 2)
        log(f"  No pushing Brier: {no_push_brier:.4f}")

        if best_push['brier'] < no_push_brier:
            log(f"  Pushing IMPROVES Brier by {no_push_brier - best_push['brier']:.4f}")
            push_config = {'enabled': True, 'strength': best_push['strength'], 'method': best_push['method']}
        else:
            log(f"  Pushing does NOT improve Brier, will skip")
            push_config = {'enabled': False, 'strength': 1.0, 'method': 'power'}

        calibration_model = best_calibration
    else:
        log("WARNING: No OOF data for calibration, using default")
        calibration_model = {'type': 'logistic', 'scale': 10, 'bias': 0}
        push_config = {'enabled': False, 'strength': 1.0, 'method': 'power'}
        all_calibrations = {}

    # Save calibration model
    with open(MODELS_DIR / "calibration.pkl", "wb") as f:
        pickle.dump({
            'best': calibration_model,
            'all': all_calibrations,
            'push_config': push_config
        }, f)
    log(f"Saved: {MODELS_DIR / 'calibration.pkl'}")

    # =========================================================================
    # LOAD SUBMISSION TEMPLATE
    # =========================================================================
    print_separator("Loading Submission Template")

    # Look for sample submission file
    sample_submissions = list(DATA_DIR.glob("*Submission*.csv")) + \
                         list(DATA_DIR.glob("*submission*.csv"))

    if sample_submissions:
        submission_template = pd.read_csv(sample_submissions[0])
        log(f"Loaded: {sample_submissions[0]}")
        log(f"Matchups to predict: {len(submission_template):,}")
    else:
        log("ERROR: No submission template found")
        sys.exit(1)

    # =========================================================================
    # BUILD FEATURES (VECTORIZED)
    # =========================================================================
    print_separator("Building Features (Vectorized)")

    submission_df = build_features_vectorized(
        submission_template, all_ratings, season_stats,
        gold_medal_features, seeds, massey_ordinals
    )

    # =========================================================================
    # GENERATE PREDICTIONS (BATCH, GENDER-SPECIFIC)
    # =========================================================================
    print_separator("Generating Predictions (Batch)")

    import xgboost as xgb

    # Ensure feature columns exist
    for col in feature_cols:
        if col not in submission_df.columns:
            submission_df[col] = 0

    log(f"Feature columns: {len(feature_cols)}")

    # Initialize margins array
    margins = np.zeros(len(submission_df))

    if genders_in_models is not None:
        # NEW FORMAT: Use gender-specific models
        for gender in genders_in_models:
            gender_mask = submission_df['Gender'] == gender
            if gender_mask.sum() == 0:
                continue

            log(f"\n  {gender} predictions ({gender_mask.sum():,} matchups):")

            X_gender = submission_df.loc[gender_mask, feature_cols].values
            X_gender = np.nan_to_num(X_gender, nan=0.0)

            # Get all models for this gender
            gender_models = {k[1]: v for k, v in models.items() if k[0] == gender}

            # Ensemble predictions from all season models for this gender
            gender_preds = []
            for season, model in gender_models.items():
                dtest = xgb.DMatrix(X_gender, feature_names=feature_cols)
                preds = model.predict(dtest)
                gender_preds.append(preds)

            # Average across models
            gender_margins = np.mean(gender_preds, axis=0)
            margins[gender_mask] = gender_margins

            log(f"    Models: {len(gender_models)}")
            log(f"    Margin range: {gender_margins.min():.2f} to {gender_margins.max():.2f}")
    else:
        # LEGACY FORMAT: Combined models
        X = submission_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)

        log(f"Feature matrix shape: {X.shape}")

        all_preds = []
        for season, model in models.items():
            dtest = xgb.DMatrix(X, feature_names=feature_cols)
            preds = model.predict(dtest)
            all_preds.append(preds)

        margins = np.mean(all_preds, axis=0)
    log(f"Ensemble margin range: {margins.min():.2f} to {margins.max():.2f}")

    # Convert to probabilities using best calibration
    probabilities = apply_calibration(margins, calibration_model)
    log(f"Base probability range: {probabilities.min():.3f} to {probabilities.max():.3f}")

    # Apply probability pushing if it helps
    if push_config['enabled']:
        probabilities_pushed = push_probabilities(
            probabilities,
            strength=push_config['strength'],
            method=push_config['method']
        )
        log(f"Pushed probability range: {probabilities_pushed.min():.3f} to {probabilities_pushed.max():.3f}")
    else:
        probabilities_pushed = probabilities.copy()
        log("Probability pushing disabled (didn't improve OOF Brier)")

    # Apply seed-based adjustments
    if 'T1_seed' in submission_df.columns and 'T2_seed' in submission_df.columns:
        probabilities_final = apply_seed_adjustments(
            probabilities_pushed,
            submission_df['T1_seed'].values,
            submission_df['T2_seed'].values
        )
        log("Applied seed-based historical adjustments")
    else:
        probabilities_final = probabilities_pushed

    submission_df['Pred'] = probabilities_final
    submission_df['Pred_base'] = probabilities  # Keep original for comparison
    submission_df['Margin'] = margins

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print_separator("Prediction Analysis")

    log(f"\nBase prediction statistics:")
    log(f"  Mean probability: {probabilities.mean():.4f}")
    log(f"  Std probability: {probabilities.std():.4f}")
    log(f"  Min/Max: {probabilities.min():.4f} / {probabilities.max():.4f}")

    log(f"\nFinal (pushed) prediction statistics:")
    log(f"  Mean probability: {probabilities_final.mean():.4f}")
    log(f"  Std probability: {probabilities_final.std():.4f}")
    log(f"  Min/Max: {probabilities_final.min():.4f} / {probabilities_final.max():.4f}")

    log(f"\nMargin statistics:")
    log(f"  Mean margin: {margins.mean():.2f}")
    log(f"  Std margin: {margins.std():.2f}")

    # Distribution comparison
    log(f"\nProbability distribution (Base vs Pushed):")
    log(f"  {'Range':<12} {'Base':>10} {'Pushed':>10}")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        base_count = ((probabilities >= thresh - 0.05) & (probabilities < thresh + 0.05)).sum()
        pushed_count = ((probabilities_final >= thresh - 0.05) & (probabilities_final < thresh + 0.05)).sum()
        log(f"  {thresh-0.05:.2f}-{thresh+0.05:.2f}:  {base_count:>10,} {pushed_count:>10,}")

    # Extreme predictions
    log(f"\nExtreme predictions (confidence indicators):")
    log(f"  Base < 0.15 or > 0.85: {((probabilities < 0.15) | (probabilities > 0.85)).sum():,}")
    log(f"  Pushed < 0.15 or > 0.85: {((probabilities_final < 0.15) | (probabilities_final > 0.85)).sum():,}")
    log(f"  Base < 0.10 or > 0.90: {((probabilities < 0.10) | (probabilities > 0.90)).sum():,}")
    log(f"  Pushed < 0.10 or > 0.90: {((probabilities_final < 0.10) | (probabilities_final > 0.90)).sum():,}")

    # =========================================================================
    # SAVE SUBMISSIONS
    # =========================================================================
    print_separator("Saving Submissions")

    # Main submission (with all adjustments)
    final_submission = submission_df[['ID', 'Pred']].copy()
    final_submission.to_csv(SUBMISSIONS_DIR / "submission.csv", index=False)
    log(f"Saved: {SUBMISSIONS_DIR / 'submission.csv'} (final with pushing + seed adj)")

    # Conservative version (no pushing)
    conservative = submission_df[['ID']].copy()
    conservative['Pred'] = probabilities
    conservative.to_csv(SUBMISSIONS_DIR / "submission_conservative.csv", index=False)
    log(f"Saved: {SUBMISSIONS_DIR / 'submission_conservative.csv'} (base calibration only)")

    # Aggressive version (extra pushing)
    aggressive_probs = push_probabilities(probabilities, strength=2.5, method='power')
    aggressive = submission_df[['ID']].copy()
    aggressive['Pred'] = aggressive_probs
    aggressive.to_csv(SUBMISSIONS_DIR / "submission_aggressive.csv", index=False)
    log(f"Saved: {SUBMISSIONS_DIR / 'submission_aggressive.csv'} (strength=2.5 pushing)")

    # Detailed version for analysis
    detailed = submission_df[['ID', 'Season', 'T1_TeamID', 'T2_TeamID', 'T1_seed', 'T2_seed', 'Margin', 'Pred_base', 'Pred']].copy()
    detailed.to_csv(SUBMISSIONS_DIR / "submission_detailed.csv", index=False)
    log(f"Saved: {SUBMISSIONS_DIR / 'submission_detailed.csv'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_separator("PHASE 5 COMPLETE - SUMMARY")

    log(f"\nSubmission generated:")
    log(f"  - Matchups: {len(submission_df):,}")
    log(f"  - Models ensembled: {len(models)}")
    log(f"  - Calibration method: {calibration_model['type']}")
    log(f"  - Calibration Brier: {calibration_model.get('brier', 'N/A')}")

    if push_config['enabled']:
        log(f"  - Probability pushing: {push_config['method']} (strength={push_config['strength']:.1f})")
    else:
        log(f"  - Probability pushing: disabled")

    log(f"\nSubmission variants:")
    log(f"  - submission.csv: Final (calibration + pushing + seed adj)")
    log(f"  - submission_conservative.csv: Base calibration only")
    log(f"  - submission_aggressive.csv: Extra aggressive pushing")

    log(f"\nFiles saved to: {SUBMISSIONS_DIR}")
    log(f"Log saved to: {LOG_FILE}")

    print_separator()

    return 0


if __name__ == "__main__":
    sys.exit(main())
