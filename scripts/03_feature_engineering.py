#!/usr/bin/env python3
"""
Phase 3: Feature Engineering
============================
Compute all features based on winning solutions (~30-35 total features).

Inputs:
- processed/regular_sym.csv
- processed/tourney_sym.csv
- processed/seeds.csv
- processed/bradley_terry_ratings.pkl
- processed/elo_ratings.pkl
- processed/glm_quality_ratings.pkl
- data/MMasseyOrdinals.csv (external ratings: KenPom, Sagarin, RPI)

Outputs:
- processed/season_stats.csv
- processed/gold_medal_features.pkl
- processed/massey_ordinals.csv
- processed/training_features.csv
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
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)

# Logging setup
LOG_FILE = OUTPUT_DIR / "03_feature_engineering.log"


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


def calculate_season_stats(games_sym):
    """
    Compute season averages for each team.
    Uses ONLY the subset of stats that winning solutions use.
    Computes separately per gender if Gender column exists.
    """
    log("  Computing season averages per team...")

    # Check if Gender column exists
    has_gender = 'Gender' in games_sym.columns
    if has_gender:
        log("  Processing Men's and Women's data separately")

    # Stats that winners actually use
    agg_dict = {}

    # Basic stats
    if 'T1_Score' in games_sym.columns:
        agg_dict['T1_Score'] = 'mean'
    if 'T1_FGA' in games_sym.columns:
        agg_dict['T1_FGA'] = 'mean'
    if 'T1_Blk' in games_sym.columns:
        agg_dict['T1_Blk'] = 'mean'
    if 'T1_PF' in games_sym.columns:
        agg_dict['T1_PF'] = 'mean'

    # Opponent stats
    if 'T2_Score' in games_sym.columns:
        agg_dict['T2_Score'] = 'mean'
    if 'T2_FGA' in games_sym.columns:
        agg_dict['T2_FGA'] = 'mean'
    if 'T2_Blk' in games_sym.columns:
        agg_dict['T2_Blk'] = 'mean'
    if 'T2_PF' in games_sym.columns:
        agg_dict['T2_PF'] = 'mean'

    # Game-level stats
    agg_dict['PointDiff'] = 'mean'
    agg_dict['T1_Win'] = ['mean', 'sum', 'count']

    if 'T1_Home' in games_sym.columns:
        agg_dict['T1_Home'] = 'mean'

    # Group by Season, TeamID (and Gender for verification, but teams are unique across genders)
    season_stats = games_sym.groupby(['Season', 'T1_TeamID']).agg(agg_dict)

    # Flatten column names
    season_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                            for col in season_stats.columns.values]

    season_stats = season_stats.reset_index()
    season_stats = season_stats.rename(columns={'T1_TeamID': 'TeamID'})

    # Rename columns for clarity
    rename_map = {
        'T1_Score_mean': 'avg_Score',
        'T1_FGA_mean': 'avg_FGA',
        'T1_Blk_mean': 'avg_Blk',
        'T1_PF_mean': 'avg_PF',
        'T2_Score_mean': 'avg_opponent_Score',
        'T2_FGA_mean': 'avg_opponent_FGA',
        'T2_Blk_mean': 'avg_opponent_Blk',
        'T2_PF_mean': 'avg_opponent_PF',
        'PointDiff_mean': 'avg_PointDiff',
        'T1_Win_mean': 'WinPct',
        'T1_Win_sum': 'Wins',
        'T1_Win_count': 'Games',
        'T1_Home_mean': 'avg_Home'
    }

    season_stats = season_stats.rename(columns=rename_map)

    log(f"  Computed stats for {len(season_stats):,} team-seasons")
    log(f"  Features: {list(season_stats.columns)}")

    return season_stats


def calculate_win_ratio_14d(games_sym):
    """
    Calculate win ratio in last 14 days of season.
    From gold-medal solution.
    """
    log("  Computing 14-day win ratio...")

    ratio_dict = {}
    games_in_window = 0

    for (season, team), group in games_sym.groupby(['Season', 'T1_TeamID']):
        max_day = group['DayNum'].max()
        last_14 = group[group['DayNum'] >= max_day - 14]

        if len(last_14) > 0:
            win_ratio = last_14['T1_Win'].mean()
            games_in_window += len(last_14)
        else:
            win_ratio = 0.5

        ratio_dict[(season, team)] = win_ratio

    log(f"  Computed for {len(ratio_dict):,} team-seasons")
    log(f"  Avg games in 14-day window: {games_in_window / len(ratio_dict):.1f}")

    return ratio_dict


def calculate_away_wins(games_sym):
    """
    Calculate if team has any away wins.
    From gold-medal solution.
    """
    log("  Computing away wins feature...")

    away_dict = {}
    teams_with_away_wins = 0

    for (season, team), group in games_sym.groupby(['Season', 'T1_TeamID']):
        # Away games where T1_Home == -1
        if 'T1_Home' in group.columns:
            away_games = group[group['T1_Home'] == -1]

            if len(away_games) > 0:
                has_away_win = 1.0 if away_games['T1_Win'].sum() > 0 else 0.0
                if has_away_win:
                    teams_with_away_wins += 1
            else:
                has_away_win = np.nan
        else:
            has_away_win = np.nan

        away_dict[(season, team)] = has_away_win

    log(f"  Computed for {len(away_dict):,} team-seasons")
    log(f"  Teams with away wins: {teams_with_away_wins:,} ({teams_with_away_wins/len(away_dict)*100:.1f}%)")

    return away_dict


def calculate_weighted_wins(games_sym):
    """
    Calculate wins weighted by day of season (later games count more).
    From hoops-i-did-it-again solution.
    """
    log("  Computing weighted wins...")

    weighted_dict = {}

    for (season, team), group in games_sym.groupby(['Season', 'T1_TeamID']):
        max_day = group['DayNum'].max()

        if max_day > 0:
            # p_day weight: 0.5 to 1.5 range
            group = group.copy()
            group['p_day'] = group['DayNum'] / max_day + 0.5

            weighted_wins = (group['T1_Win'] * group['p_day']).sum()
        else:
            weighted_wins = group['T1_Win'].sum()

        weighted_dict[(season, team)] = weighted_wins

    log(f"  Computed for {len(weighted_dict):,} team-seasons")

    return weighted_dict


def load_massey_ordinals():
    """
    Load Massey Ordinals (external ratings like KenPom, Sagarin, RPI).
    Extract end-of-season ratings for key systems.
    """
    massey_path = DATA_DIR / "MMasseyOrdinals.csv"

    if not massey_path.exists():
        log("WARNING: MMasseyOrdinals.csv not found")
        return pd.DataFrame()

    log("  Loading Massey Ordinals...")

    # Key rating systems to extract
    SYSTEMS = ['POM', 'SAG', 'RPI', 'MOR', 'DOL', 'COL']  # KenPom, Sagarin, RPI, etc.

    # Read only the systems we need (file is huge)
    chunks = []
    for chunk in pd.read_csv(massey_path, chunksize=500000):
        filtered = chunk[chunk['SystemName'].isin(SYSTEMS)]
        chunks.append(filtered)

    if not chunks:
        log("  No matching systems found")
        return pd.DataFrame()

    massey = pd.concat(chunks, ignore_index=True)
    log(f"  Loaded {len(massey):,} ratings from {massey['SystemName'].nunique()} systems")

    # Get end-of-season rating for each team-season-system
    # Use the latest day number available for each season
    end_of_season = massey.loc[massey.groupby(['Season', 'TeamID', 'SystemName'])['RankingDayNum'].idxmax()]

    log(f"  End-of-season ratings: {len(end_of_season):,}")

    # Pivot to get one row per team-season with columns for each system
    ordinals = end_of_season.pivot_table(
        index=['Season', 'TeamID'],
        columns='SystemName',
        values='OrdinalRank',
        aggfunc='first'
    ).reset_index()

    # Rename columns for clarity
    ordinals.columns.name = None
    rename_map = {sys: f'Rank_{sys}' for sys in SYSTEMS if sys in ordinals.columns}
    ordinals = ordinals.rename(columns=rename_map)

    log(f"  Final ordinals shape: {ordinals.shape}")
    log(f"  Systems available: {[c for c in ordinals.columns if c.startswith('Rank_')]}")

    return ordinals


def build_features_for_matchup(season, team1, team2,
                                bt_strengths, elo_ratings, glm_quality,
                                season_stats, win_ratio_14d, away_wins, weighted_wins,
                                seeds_df, massey_ordinals=None):
    """
    Create complete feature vector for a matchup.
    """
    f = {}

    # EASY FEATURES
    f['men_women'] = 1 if team1 < 2000 else 0

    if len(seeds_df) > 0:
        seed_lookup = seeds_df[seeds_df['Season'] == season].set_index('TeamID')
        if 'SeedNum' in seed_lookup.columns:
            seed_lookup = seed_lookup['SeedNum'].to_dict()
        else:
            seed_lookup = {}
    else:
        seed_lookup = {}

    f['T1_seed'] = seed_lookup.get(team1, 16)
    f['T2_seed'] = seed_lookup.get(team2, 16)
    f['Seed_diff'] = f['T2_seed'] - f['T1_seed']

    # HARD FEATURES - Strength Ratings
    bt = bt_strengths.get(season, {})
    f['T1_bt'] = bt.get(team1, 0)
    f['T2_bt'] = bt.get(team2, 0)
    f['bt_diff'] = f['T1_bt'] - f['T2_bt']

    elo = elo_ratings.get(season, {})
    f['T1_elo'] = elo.get(team1, 1500)
    f['T2_elo'] = elo.get(team2, 1500)
    f['elo_diff'] = f['T1_elo'] - f['T2_elo']

    # HARDEST FEATURES - GLM Quality
    glm = glm_quality.get(season, {})
    f['T1_quality'] = glm.get(team1, 0)
    f['T2_quality'] = glm.get(team2, 0)
    f['quality_diff'] = f['T1_quality'] - f['T2_quality']

    # MEDIUM FEATURES - Season Averages
    stats = season_stats[season_stats['Season'] == season].set_index('TeamID')

    stat_cols = ['avg_Score', 'avg_FGA', 'avg_Blk', 'avg_PF',
                 'avg_opponent_Score', 'avg_opponent_FGA', 'avg_opponent_Blk', 'avg_opponent_PF',
                 'avg_PointDiff', 'avg_Home', 'WinPct']

    for col in stat_cols:
        if col in stats.columns:
            f[f'T1_{col}'] = stats.loc[team1, col] if team1 in stats.index else 0
            f[f'T2_{col}'] = stats.loc[team2, col] if team2 in stats.index else 0
        else:
            f[f'T1_{col}'] = 0
            f[f'T2_{col}'] = 0

    # GOLD-MEDAL FEATURES
    f['T1_WinRatio14d'] = win_ratio_14d.get((season, team1), 0.5)
    f['T2_WinRatio14d'] = win_ratio_14d.get((season, team2), 0.5)
    f['T1_away_wins'] = away_wins.get((season, team1), np.nan)
    f['T2_away_wins'] = away_wins.get((season, team2), np.nan)

    # HOOPS FEATURES
    f['T1_weighted_wins'] = weighted_wins.get((season, team1), 0)
    f['T2_weighted_wins'] = weighted_wins.get((season, team2), 0)

    # MASSEY ORDINALS (External ratings: KenPom, Sagarin, RPI, etc.)
    if massey_ordinals is not None and len(massey_ordinals) > 0:
        season_ordinals = massey_ordinals[massey_ordinals['Season'] == season].set_index('TeamID')
        rank_cols = [c for c in massey_ordinals.columns if c.startswith('Rank_')]

        for col in rank_cols:
            # Lower rank = better team, so we want T2 - T1 for consistency with seed_diff
            t1_rank = season_ordinals.loc[team1, col] if team1 in season_ordinals.index else 200
            t2_rank = season_ordinals.loc[team2, col] if team2 in season_ordinals.index else 200

            # Handle NaN
            if pd.isna(t1_rank):
                t1_rank = 200
            if pd.isna(t2_rank):
                t2_rank = 200

            f[f'T1_{col}'] = t1_rank
            f[f'T2_{col}'] = t2_rank
            f[f'{col}_diff'] = t2_rank - t1_rank  # Positive = T1 better

    return f


def main():
    """Main execution function."""
    # Clear previous log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    print_separator("PHASE 3: FEATURE ENGINEERING")

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print_separator("Loading Processed Data")

    # Load symmetric games
    regular_sym_path = PROCESSED_DIR / "regular_sym.csv"
    tourney_sym_path = PROCESSED_DIR / "tourney_sym.csv"
    seeds_path = PROCESSED_DIR / "seeds.csv"

    if not regular_sym_path.exists():
        log(f"ERROR: {regular_sym_path} not found. Run 01_data_preparation.py first.")
        sys.exit(1)

    regular_sym = pd.read_csv(regular_sym_path)
    log(f"Loaded regular_sym: {len(regular_sym):,} records")

    if tourney_sym_path.exists():
        tourney_sym = pd.read_csv(tourney_sym_path)
        log(f"Loaded tourney_sym: {len(tourney_sym):,} records")
    else:
        log("WARNING: tourney_sym.csv not found")
        tourney_sym = pd.DataFrame()

    if seeds_path.exists():
        seeds = pd.read_csv(seeds_path)
        log(f"Loaded seeds: {len(seeds):,} entries")
    else:
        log("WARNING: seeds.csv not found")
        seeds = pd.DataFrame()

    # Load ratings
    bt_path = PROCESSED_DIR / "bradley_terry_ratings.pkl"
    elo_path = PROCESSED_DIR / "elo_ratings.pkl"
    glm_path = PROCESSED_DIR / "glm_quality_ratings.pkl"

    if not bt_path.exists():
        log(f"ERROR: {bt_path} not found. Run 02_team_strength_ratings.py first.")
        sys.exit(1)

    with open(bt_path, "rb") as f:
        bt_data = pickle.load(f)
        bt_strengths = bt_data['ratings']
    log(f"Loaded Bradley-Terry ratings for {len(bt_strengths)} seasons")

    with open(elo_path, "rb") as f:
        elo_ratings = pickle.load(f)
    log(f"Loaded Elo ratings for {len(elo_ratings)} seasons")

    with open(glm_path, "rb") as f:
        glm_quality = pickle.load(f)
    log(f"Loaded GLM Quality ratings for {len(glm_quality)} seasons")

    seasons = sorted(regular_sym['Season'].unique())
    log(f"\nSeasons to process: {seasons}")

    # =========================================================================
    # COMPUTE SEASON STATISTICS
    # =========================================================================
    print_separator("Computing Season Statistics")

    season_stats = calculate_season_stats(regular_sym)
    season_stats.to_csv(PROCESSED_DIR / "season_stats.csv", index=False)
    log(f"Saved: {PROCESSED_DIR / 'season_stats.csv'}")

    # Show sample statistics
    log("\nSample season statistics (first 5 teams of most recent season):")
    latest_season = max(seasons)
    sample = season_stats[season_stats['Season'] == latest_season].head()
    for _, row in sample.iterrows():
        log(f"  Team {row['TeamID']}: {row['Wins']:.0f}W, {row['WinPct']*100:.1f}%, Avg Margin: {row['avg_PointDiff']:.1f}")

    # =========================================================================
    # LOAD MASSEY ORDINALS (External Ratings)
    # =========================================================================
    print_separator("Loading Massey Ordinals (KenPom, Sagarin, RPI)")

    massey_ordinals = load_massey_ordinals()

    if len(massey_ordinals) > 0:
        massey_ordinals.to_csv(PROCESSED_DIR / "massey_ordinals.csv", index=False)
        log(f"Saved: {PROCESSED_DIR / 'massey_ordinals.csv'}")

        # Show coverage
        log(f"\nMassey Ordinals coverage:")
        for col in [c for c in massey_ordinals.columns if c.startswith('Rank_')]:
            coverage = massey_ordinals[col].notna().sum()
            log(f"  {col}: {coverage:,} team-seasons")

    # =========================================================================
    # COMPUTE GOLD-MEDAL FEATURES
    # =========================================================================
    print_separator("Computing Gold-Medal Features")

    win_ratio_14d = calculate_win_ratio_14d(regular_sym)
    away_wins = calculate_away_wins(regular_sym)
    weighted_wins = calculate_weighted_wins(regular_sym)

    # Save gold-medal features
    gold_medal_features = {
        'win_ratio_14d': win_ratio_14d,
        'away_wins': away_wins,
        'weighted_wins': weighted_wins
    }

    with open(PROCESSED_DIR / "gold_medal_features.pkl", "wb") as f:
        pickle.dump(gold_medal_features, f)
    log(f"Saved: {PROCESSED_DIR / 'gold_medal_features.pkl'}")

    # =========================================================================
    # BUILD TRAINING FEATURES (FROM TOURNAMENT GAMES)
    # =========================================================================
    print_separator("Building Training Features from Tournament Games")

    if len(tourney_sym) == 0:
        log("WARNING: No tournament data available for training features")
        training_rows = []
    else:
        training_rows = []
        processed = 0

        # Get unique tournament matchups (one perspective per game)
        tourney_games = tourney_sym[tourney_sym['T1_Win'] == 1].copy()

        log(f"Processing {len(tourney_games):,} tournament games...")

        for _, row in tourney_games.iterrows():
            season = row['Season']
            team1 = row['T1_TeamID']
            team2 = row['T2_TeamID']
            point_diff = row['PointDiff']

            features = build_features_for_matchup(
                season, team1, team2,
                bt_strengths, elo_ratings, glm_quality,
                season_stats, win_ratio_14d, away_wins, weighted_wins,
                seeds, massey_ordinals
            )

            # Get gender from row or infer from team ID
            gender = row.get('Gender', 'M' if team1 < 2000 else 'W')

            features['Season'] = season
            features['T1_TeamID'] = team1
            features['T2_TeamID'] = team2
            features['Gender'] = gender
            features['PointDiff'] = point_diff
            features['T1_Win'] = 1

            training_rows.append(features)

            # Also add reverse perspective
            features_rev = build_features_for_matchup(
                season, team2, team1,
                bt_strengths, elo_ratings, glm_quality,
                season_stats, win_ratio_14d, away_wins, weighted_wins,
                seeds, massey_ordinals
            )

            features_rev['Season'] = season
            features_rev['T1_TeamID'] = team2
            features_rev['T2_TeamID'] = team1
            features_rev['Gender'] = gender
            features_rev['PointDiff'] = -point_diff
            features_rev['T1_Win'] = 0

            training_rows.append(features_rev)

            processed += 1

            if processed % 500 == 0:
                log(f"  Processed {processed:,} games...")

    training_df = pd.DataFrame(training_rows)

    if len(training_df) > 0:
        training_df.to_csv(PROCESSED_DIR / "training_features.csv", index=False)
        log(f"Saved: {PROCESSED_DIR / 'training_features.csv'}")

        # =========================================================================
        # FEATURE SUMMARY
        # =========================================================================
        print_separator("Feature Summary")

        feature_cols = [c for c in training_df.columns
                        if c not in ['Season', 'T1_TeamID', 'T2_TeamID', 'PointDiff', 'T1_Win', 'Gender']]

        log(f"\nTotal features: {len(feature_cols)}")
        log(f"\nFeature list:")
        for i, col in enumerate(sorted(feature_cols), 1):
            log(f"  {i:2d}. {col}")

        # Feature statistics (numeric only)
        log(f"\nFeature statistics:")
        numeric_feature_cols = [c for c in feature_cols if training_df[c].dtype in ['int64', 'float64']]
        for col in numeric_feature_cols[:20]:  # Limit to 20 for brevity
            data = training_df[col].dropna()
            if len(data) > 0:
                log(f"  {col}: mean={data.mean():.3f}, std={data.std():.3f}, "
                    f"min={data.min():.3f}, max={data.max():.3f}, "
                    f"missing={training_df[col].isna().sum()}")

        # Correlation with target (numeric features only)
        log(f"\nCorrelation with PointDiff (top 10):")
        corr_cols = [c for c in numeric_feature_cols if c in training_df.columns]
        correlations = training_df[corr_cols + ['PointDiff']].corr()['PointDiff'].drop('PointDiff')
        correlations = correlations.abs().sort_values(ascending=False)
        for col, corr in correlations.head(10).items():
            log(f"  {col}: {corr:.3f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_separator("PHASE 3 COMPLETE - SUMMARY")

    log(f"\nFeature engineering completed:")
    log(f"  - Season statistics computed for {len(season_stats):,} team-seasons")
    log(f"  - Gold-medal features computed")
    log(f"  - Training features: {len(training_df):,} samples")

    if len(training_df) > 0:
        log(f"  - Feature count: {len(feature_cols)}")
        log(f"  - Seasons in training: {sorted(training_df['Season'].unique())}")

    log(f"\nOutputs saved to: {PROCESSED_DIR}")
    log(f"Log saved to: {LOG_FILE}")

    print_separator()

    return 0


if __name__ == "__main__":
    sys.exit(main())
