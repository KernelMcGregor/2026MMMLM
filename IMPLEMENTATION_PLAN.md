# March Machine Learning Mania 2026 - Implementation Plan

## Overview

This plan is based on analysis of **all winning solutions** from recent competitions. It uses **Bradley-Terry** as the core team strength model (replacing goto_conversion), integrated with XGBoost for final predictions.

---

## Key Findings From Winning Solutions

| Finding | Implication |
|---------|-------------|
| Simplest approaches won (baseline + overrides) | Don't over-engineer |
| Most use only ~25-34 features | Quality > quantity |
| Elo + GLM Quality are the "hard" features | Must include these |
| Last N games, EMA, Four Factors NOT used | Remove from plan |
| Late-season weighting IS used | Add WinRatio14d, weighted games |
| Away performance matters | Add away_wins features |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: DATA PREPARATION                     │
│  Load Kaggle data, clean, merge men's/women's, normalize overtime       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: TEAM STRENGTH RATINGS                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐            │
│  │  Bradley-Terry  │ │      Elo        │ │   GLM Quality   │            │
│  │  (uses margin)  │ │  (uses W/L)     │ │  (uses margin)  │            │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: FEATURE ENGINEERING                        │
│  ~30-35 features based on what winning solutions actually use           │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PHASE 4: MODEL TRAINING                          │
│  XGBoost predicting point differential with LOSO cross-validation       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PHASE 5: CALIBRATION & SUBMISSION                │
│  Spline calibration → probabilities → optional strategic overrides      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Preparation

### 1.1 Load and Combine Data

```python
import pandas as pd
import numpy as np
from collections import defaultdict

DATA_DIR = "path/to/kaggle/data"

# Load all data
m_regular = pd.read_csv(f"{DATA_DIR}/MRegularSeasonDetailedResults.csv")
w_regular = pd.read_csv(f"{DATA_DIR}/WRegularSeasonDetailedResults.csv")
m_tourney = pd.read_csv(f"{DATA_DIR}/MNCAATourneyDetailedResults.csv")
w_tourney = pd.read_csv(f"{DATA_DIR}/WNCAATourneyDetailedResults.csv")
m_seeds = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
w_seeds = pd.read_csv(f"{DATA_DIR}/WNCAATourneySeeds.csv")

# Combine men's and women's
regular_results = pd.concat([m_regular, w_regular], ignore_index=True)
tourney_results = pd.concat([m_tourney, w_tourney], ignore_index=True)
seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)
```

### 1.2 Normalize Overtime & Add Home Court

```python
def normalize_overtime(df):
    """Normalize stats to 40-minute game equivalent"""
    df = df.copy()
    ot_factor = 40 / (40 + 5 * df['NumOT'])

    stat_cols = ['WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
                 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
                 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

    for col in stat_cols:
        if col in df.columns:
            df[col] = df[col] * ot_factor
    return df

# Add home court indicator (used by gold-medal solution)
wloc = {'H': 1, 'A': -1, 'N': np.nan}
regular_results['WHome'] = regular_results['WLoc'].map(lambda x: wloc[x])

regular_results = normalize_overtime(regular_results)
tourney_results = normalize_overtime(tourney_results)
```

### 1.3 Create Symmetric Game DataFrame

```python
def create_symmetric_games(df):
    """Each game appears from both team perspectives"""

    # Winner perspective
    df1 = df.copy()
    df1['T1_TeamID'] = df['WTeamID']
    df1['T2_TeamID'] = df['LTeamID']
    df1['T1_Score'] = df['WScore']
    df1['T2_Score'] = df['LScore']
    df1['PointDiff'] = df1['T1_Score'] - df1['T2_Score']
    df1['T1_Win'] = 1
    df1['T1_Home'] = df.get('WHome', np.nan)
    df1['T2_Home'] = -df.get('WHome', np.nan)  # Opposite

    for prefix in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
        if f'W{prefix}' in df.columns:
            df1[f'T1_{prefix}'] = df[f'W{prefix}']
            df1[f'T2_{prefix}'] = df[f'L{prefix}']

    # Loser perspective (swap everything)
    df2 = df.copy()
    df2['T1_TeamID'] = df['LTeamID']
    df2['T2_TeamID'] = df['WTeamID']
    df2['T1_Score'] = df['LScore']
    df2['T2_Score'] = df['WScore']
    df2['PointDiff'] = df2['T1_Score'] - df2['T2_Score']
    df2['T1_Win'] = 0
    df2['T1_Home'] = -df.get('WHome', np.nan)
    df2['T2_Home'] = df.get('WHome', np.nan)

    for prefix in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
        if f'W{prefix}' in df.columns:
            df2[f'T1_{prefix}'] = df[f'L{prefix}']
            df2[f'T2_{prefix}'] = df[f'W{prefix}']

    return pd.concat([df1, df2], ignore_index=True)

regular_sym = create_symmetric_games(regular_results)
tourney_sym = create_symmetric_games(tourney_results)
```

### 1.4 Parse Seeds

```python
def parse_seed(seed_str):
    """Extract numeric seed from seed string (e.g., 'W01' -> 1)"""
    return int(seed_str[1:3])

seeds['SeedNum'] = seeds['Seed'].apply(parse_seed)
```

---

## Phase 2: Team Strength Ratings

### 2.1 Bradley-Terry with Margin (Primary)

```python
from sklearn.linear_model import Ridge

def fit_bradley_terry_margin(games_df, teams=None):
    """
    Fit Bradley-Terry model using point differential.
    Model: Margin = Strength_winner - Strength_loser + HomeAdvantage + noise

    Returns:
        strengths: dict {team_id: strength}
        home_advantage: float
    """
    if teams is None:
        teams = list(set(games_df['WTeamID'].tolist() + games_df['LTeamID'].tolist()))

    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Build design matrix
    X = np.zeros((len(games_df), n_teams + 1))  # +1 for home advantage
    y = np.zeros(len(games_df))

    for i, (_, row) in enumerate(games_df.iterrows()):
        w_idx = team_to_idx.get(row['WTeamID'])
        l_idx = team_to_idx.get(row['LTeamID'])

        if w_idx is not None and l_idx is not None:
            X[i, w_idx] = 1
            X[i, l_idx] = -1
            y[i] = row['WScore'] - row['LScore']

            if 'WLoc' in row:
                if row['WLoc'] == 'H':
                    X[i, -1] = 1
                elif row['WLoc'] == 'A':
                    X[i, -1] = -1

    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(X, y)

    strengths = model.coef_[:n_teams]
    strengths = strengths - strengths.mean()

    return {teams[i]: strengths[i] for i in range(n_teams)}, model.coef_[-1]
```

### 2.2 Elo Rating System

```python
def calculate_elo_ratings(games_df, k_factor=32, home_advantage=100, initial_elo=1500):
    """
    Calculate Elo ratings from game results.
    Games must be sorted by DayNum.
    """
    elo = defaultdict(lambda: initial_elo)
    games_df = games_df.sort_values('DayNum')

    for _, row in games_df.iterrows():
        winner = row['WTeamID']
        loser = row['LTeamID']

        w_elo = elo[winner]
        l_elo = elo[loser]

        # Home court adjustment
        if row.get('WLoc') == 'H':
            w_elo_adj = w_elo + home_advantage
            l_elo_adj = l_elo
        elif row.get('WLoc') == 'A':
            w_elo_adj = w_elo
            l_elo_adj = l_elo + home_advantage
        else:
            w_elo_adj, l_elo_adj = w_elo, l_elo

        exp_w = 1 / (1 + 10 ** ((l_elo_adj - w_elo_adj) / 400))

        elo[winner] += k_factor * (1 - exp_w)
        elo[loser] -= k_factor * (1 - exp_w)

    return dict(elo)
```

### 2.3 GLM Quality Score

```python
import statsmodels.api as sm

def fit_glm_quality(games_sym, tourney_teams=None):
    """
    Fit GLM: PointDiff ~ Team_strength - Opponent_strength
    Coefficients ARE the team strengths (opponent-adjusted).
    """
    df = games_sym.copy()

    if tourney_teams is not None:
        mask = df['T1_TeamID'].isin(tourney_teams) | df['T2_TeamID'].isin(tourney_teams)
        df = df[mask].copy()

    df['T1_str'] = df['T1_TeamID'].astype(str)
    df['T2_str'] = df['T2_TeamID'].astype(str)

    formula = "PointDiff ~ -1 + C(T1_str) + C(T2_str)"
    model = sm.GLM.from_formula(formula=formula, data=df, family=sm.families.Gaussian()).fit()

    quality = {}
    for param, value in model.params.items():
        if 'T1_str' in param:
            team_id = int(param.split('[')[1].split(']')[0].replace('T.', ''))
            quality[team_id] = value

    mean_q = np.mean(list(quality.values()))
    return {k: v - mean_q for k, v in quality.items()}
```

---

## Phase 3: Feature Engineering

### 3.1 Features Actually Used by Winners (~30-35 total)

Based on analysis of all example notebooks:

```python
FEATURES = {
    # EASY FEATURES (4)
    'easy': [
        'men_women',      # 1=men, 0=women
        'T1_seed',        # Tournament seed
        'T2_seed',
        'Seed_diff',      # T2_seed - T1_seed (positive = T1 favored)
    ],

    # MEDIUM FEATURES - Season Averages (16)
    'medium': [
        'T1_avg_Score', 'T2_avg_Score',
        'T1_avg_FGA', 'T2_avg_FGA',
        'T1_avg_Blk', 'T2_avg_Blk',
        'T1_avg_PF', 'T2_avg_PF',
        'T1_avg_opponent_FGA', 'T2_avg_opponent_FGA',
        'T1_avg_opponent_Blk', 'T2_avg_opponent_Blk',
        'T1_avg_opponent_PF', 'T2_avg_opponent_PF',
        'T1_avg_PointDiff', 'T2_avg_PointDiff',
    ],

    # HARD FEATURES - Strength Ratings (5)
    'hard': [
        'T1_elo', 'T2_elo', 'elo_diff',
        'T1_bt', 'T2_bt',  # Bradley-Terry (our addition)
    ],

    # HARDEST FEATURES - GLM Quality (2)
    'hardest': [
        'T1_quality', 'T2_quality',
    ],

    # GOLD-MEDAL BONUS FEATURES (6)
    'gold_medal': [
        'T1_WinRatio14d', 'T2_WinRatio14d',  # Win rate in last 14 days
        'T1_away_wins', 'T2_away_wins',       # Has away wins (binary)
        'T1_avg_Home', 'T2_avg_Home',         # Home game proportion
    ],

    # HOOPS BONUS FEATURES (2) - Optional
    'hoops': [
        'T1_weighted_wins', 'T2_weighted_wins',  # Late-season weighted
        # 'T1_opp_wins', 'T2_opp_wins',          # Opponent SoS
    ],
}
```

### 3.2 Compute Season Averages

```python
def calculate_season_stats(games_sym):
    """
    Compute season averages for each team.
    Uses ONLY the subset of stats that winning solutions use.
    """
    # Stats that winners actually use
    keep_cols = ['T1_Score', 'T1_FGA', 'T1_Blk', 'T1_PF',
                 'T2_Score', 'T2_FGA', 'T2_Blk', 'T2_PF',
                 'T1_Home', 'PointDiff']

    agg_cols = [c for c in keep_cols if c in games_sym.columns]

    season_avgs = games_sym.groupby(['Season', 'T1_TeamID'])[agg_cols].mean()
    season_avgs = season_avgs.reset_index().rename(columns={'T1_TeamID': 'TeamID'})

    # Rename to indicate these are averages
    rename_dict = {c: f'avg_{c}' for c in agg_cols}
    season_avgs = season_avgs.rename(columns=rename_dict)

    return season_avgs
```

### 3.3 Compute Gold-Medal Features

```python
def calculate_win_ratio_14d(games_sym):
    """
    Calculate win ratio in last 14 days of season.
    From gold-medal solution.
    """
    ratio_dict = {}

    for (season, team), group in games_sym.groupby(['Season', 'T1_TeamID']):
        max_day = group['DayNum'].max()
        last_14 = group[group['DayNum'] >= max_day - 14]

        if len(last_14) > 0:
            win_ratio = last_14['T1_Win'].mean()
        else:
            win_ratio = 0.5

        ratio_dict[(season, team)] = win_ratio

    return ratio_dict

def calculate_away_wins(games_sym):
    """
    Calculate if team has any away wins.
    From gold-medal solution.
    """
    away_dict = {}

    for (season, team), group in games_sym.groupby(['Season', 'T1_TeamID']):
        # Away games where T1_Home == -1
        away_games = group[group['T1_Home'] == -1]

        if len(away_games) > 0:
            has_away_win = 1.0 if away_games['T1_Win'].sum() > 0 else 0.0
        else:
            has_away_win = np.nan

        away_dict[(season, team)] = has_away_win

    return away_dict
```

### 3.4 Compute Weighted Wins (Hoops Feature)

```python
def calculate_weighted_wins(games_sym):
    """
    Calculate wins weighted by day of season (later games count more).
    From hoops-i-did-it-again solution.
    """
    weighted_dict = {}

    for (season, team), group in games_sym.groupby(['Season', 'T1_TeamID']):
        max_day = group['DayNum'].max()

        # p_day weight: 0.5 to 1.5 range
        group = group.copy()
        group['p_day'] = group['DayNum'] / max_day + 0.5

        weighted_wins = (group['T1_Win'] * group['p_day']).sum()
        weighted_dict[(season, team)] = weighted_wins

    return weighted_dict
```

### 3.5 Build Complete Feature Set

```python
def build_features(season, team1, team2,
                   bt_strengths, elo_ratings, glm_quality,
                   season_stats, win_ratio_14d, away_wins, weighted_wins,
                   seeds_df):
    """
    Create complete feature vector for a matchup.
    """
    f = {}

    # EASY FEATURES
    f['men_women'] = 1 if team1 < 2000 else 0

    seed_lookup = seeds_df[seeds_df['Season'] == season].set_index('TeamID')['SeedNum'].to_dict()
    f['T1_seed'] = seed_lookup.get(team1, 16)
    f['T2_seed'] = seed_lookup.get(team2, 16)
    f['Seed_diff'] = f['T2_seed'] - f['T1_seed']

    # HARD FEATURES - Strength Ratings
    bt = bt_strengths.get(season, {})
    f['T1_bt'] = bt.get(team1, 0)
    f['T2_bt'] = bt.get(team2, 0)

    elo = elo_ratings.get(season, {})
    f['T1_elo'] = elo.get(team1, 1500)
    f['T2_elo'] = elo.get(team2, 1500)
    f['elo_diff'] = f['T1_elo'] - f['T2_elo']

    # HARDEST FEATURES - GLM Quality
    glm = glm_quality.get(season, {})
    f['T1_quality'] = glm.get(team1, 0)
    f['T2_quality'] = glm.get(team2, 0)

    # MEDIUM FEATURES - Season Averages
    stats = season_stats[season_stats['Season'] == season].set_index('TeamID')

    for col in ['avg_T1_Score', 'avg_T1_FGA', 'avg_T1_Blk', 'avg_T1_PF',
                'avg_T2_Score', 'avg_T2_FGA', 'avg_T2_Blk', 'avg_T2_PF',
                'avg_PointDiff', 'avg_T1_Home']:
        t1_col = col.replace('T1_', '').replace('T2_', 'opponent_')
        t2_col = col.replace('T2_', '').replace('T1_', 'opponent_')

        if team1 in stats.index and col in stats.columns:
            f[f'T1_{t1_col}'] = stats.loc[team1, col]
        else:
            f[f'T1_{t1_col}'] = 0

        if team2 in stats.index and col in stats.columns:
            f[f'T2_{t2_col}'] = stats.loc[team2, col]
        else:
            f[f'T2_{t2_col}'] = 0

    # GOLD-MEDAL FEATURES
    f['T1_WinRatio14d'] = win_ratio_14d.get((season, team1), 0.5)
    f['T2_WinRatio14d'] = win_ratio_14d.get((season, team2), 0.5)
    f['T1_away_wins'] = away_wins.get((season, team1), np.nan)
    f['T2_away_wins'] = away_wins.get((season, team2), np.nan)

    # HOOPS FEATURES
    f['T1_weighted_wins'] = weighted_wins.get((season, team1), 0)
    f['T2_weighted_wins'] = weighted_wins.get((season, team2), 0)

    return f
```

---

## Phase 4: Model Training

### 4.1 XGBoost Parameters (from winning solutions)

```python
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
    'seed': 42
}

NUM_BOOST_ROUNDS = 700
EARLY_STOPPING_ROUNDS = 50
```

### 4.2 Leave-One-Season-Out CV

```python
import xgboost as xgb

def train_loso_models(X, y, feature_cols, params, num_rounds):
    """
    Train XGBoost with Leave-One-Season-Out cross-validation.
    """
    seasons = sorted(X['Season'].unique())
    models = {}
    oof_predictions = []
    oof_targets = []
    oof_seasons = []

    for holdout_season in seasons:
        print(f"Training with holdout season {holdout_season}...")

        train_mask = X['Season'] != holdout_season
        val_mask = X['Season'] == holdout_season

        X_train = X.loc[train_mask, feature_cols].values
        y_train = y[train_mask].values
        X_val = X.loc[val_mask, feature_cols].values
        y_val = y[val_mask].values

        if len(X_val) == 0:
            continue

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            evals=[(dval, 'val')],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False
        )

        models[holdout_season] = model

        preds = model.predict(dval)
        oof_predictions.extend(preds)
        oof_targets.extend(y_val)
        oof_seasons.extend([holdout_season] * len(preds))

    return models, np.array(oof_predictions), np.array(oof_targets), np.array(oof_seasons)
```

---

## Phase 5: Calibration & Submission

### 5.1 Spline Calibration

```python
from scipy.interpolate import UnivariateSpline

def fit_spline_calibration(predictions, targets, clip_range=25):
    """Convert point differential predictions to win probabilities."""
    binary_targets = (targets > 0).astype(float)

    sorted_data = sorted(zip(predictions, binary_targets))
    sorted_preds, sorted_targets = zip(*sorted_data)

    clipped_preds = np.clip(sorted_preds, -clip_range, clip_range)

    spline = UnivariateSpline(
        clipped_preds,
        sorted_targets,
        k=5,
        s=len(sorted_preds) * 0.1
    )

    return spline

def predict_probability(margin, spline_model, clip_range=25):
    """Convert margin to probability."""
    clipped = np.clip(margin, -clip_range, clip_range)
    prob = spline_model(clipped)
    return np.clip(prob, 0.01, 0.99)
```

### 5.2 Generate Submission

```python
def generate_submission(submission_df, models, spline_model,
                        bt_strengths, elo_ratings, glm_quality,
                        season_stats, win_ratio_14d, away_wins, weighted_wins,
                        seeds_df, feature_cols):
    """Generate predictions for all matchups."""

    submission_df = submission_df.copy()
    parts = submission_df['ID'].str.split('_', expand=True)
    submission_df['Season'] = parts[0].astype(int)
    submission_df['T1_TeamID'] = parts[1].astype(int)
    submission_df['T2_TeamID'] = parts[2].astype(int)

    predictions = []

    for idx, row in submission_df.iterrows():
        features = build_features(
            row['Season'], row['T1_TeamID'], row['T2_TeamID'],
            bt_strengths, elo_ratings, glm_quality,
            season_stats, win_ratio_14d, away_wins, weighted_wins,
            seeds_df
        )

        X = np.array([[features.get(col, 0) for col in feature_cols]])

        # Ensemble all LOSO models
        margin_preds = []
        for model in models.values():
            dtest = xgb.DMatrix(X, feature_names=feature_cols)
            margin_preds.append(model.predict(dtest)[0])

        avg_margin = np.mean(margin_preds)
        prob = predict_probability(avg_margin, spline_model)
        predictions.append(prob)

    submission_df['Pred'] = predictions
    return submission_df[['ID', 'Pred']]
```

### 5.3 Optional: Confidence Boosting

From final-solution-ncaa-2025:

```python
def apply_confidence_boost(predictions, threshold=0.85, boost=0.1):
    """
    Increase confidence on predictions below threshold.
    Winners often pushed predictions toward extremes.
    """
    boosted = predictions.copy()
    mask = predictions < threshold
    boosted[mask] = predictions[mask] + predictions[mask] * boost
    return np.clip(boosted, 0.01, 0.99)
```

---

## Phase 6: Strategic Overrides (Optional, High Risk)

### 6.1 The 33% Rule

From winning solutions: optimal expected return when betting on upsets with ~33% probability.

```python
def apply_strategic_override(submission_df, team_id, max_round, seeds_df):
    """
    Override predictions for a specific team to win through max_round.

    Args:
        team_id: Team to bet on (e.g., 1196 for Florida)
        max_round: How far team goes (2=Round of 32, 6=Final Four, 7=Champion)
    """
    # Implementation from ncaa2025-3th-place-solution
    # Sets Pred=1.0 for team to win all games up to max_round
    pass
```

### 6.2 Decorrelation Strategy

From APPROACH.md:
- Don't bet on popular upsets
- Bet on ONE favorite AND ONE underdog (decorrelates from both strategies)
- Position in sparse outcome space

---

## Summary: Final Feature List (~30 features)

| Category | Features | Source |
|----------|----------|--------|
| **Easy** | men_women, T1_seed, T2_seed, Seed_diff | All solutions |
| **Medium** | avg_Score, avg_FGA, avg_Blk, avg_PF (both teams + opponents) | vilnius, gold-medal |
| **Medium** | avg_PointDiff | All ML solutions |
| **Hard** | T1_elo, T2_elo, elo_diff | vilnius, gold-medal |
| **Hard** | T1_bt, T2_bt (Bradley-Terry) | Our addition |
| **Hardest** | T1_quality, T2_quality (GLM) | vilnius, gold-medal |
| **Gold-Medal** | WinRatio14d, away_wins, avg_Home | gold-medal solution |
| **Hoops** | weighted_wins | hoops solution |

---

## What Was Removed (Not Used by Winners)

| Feature | Why Removed |
|---------|-------------|
| Last N games stats | Not used by any winning solution |
| EMA (exponential moving average) | Not used |
| Four Factors (eFG%, TO rate, ORB rate, FT rate) | Not used |
| Pace/Efficiency | Not used |
| Explicit opponent adjustment | GLM Quality handles this implicitly |

---

## Expected Performance

| Metric | Target |
|--------|--------|
| OOF MAE (margin) | 8.5 - 9.5 points |
| OOF Brier Score | 0.160 - 0.175 |
| Feature count | ~30 |
