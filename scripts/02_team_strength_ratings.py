#!/usr/bin/env python3
"""
Phase 2: Team Strength Ratings
==============================
Compute Bradley-Terry, Elo, and GLM Quality ratings for all teams.

Inputs:
- processed/regular_sym.csv
- processed/seeds.csv

Outputs:
- processed/bradley_terry_ratings.csv
- processed/elo_ratings.csv
- processed/glm_quality_ratings.csv
- processed/all_ratings.csv (combined)
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)

# Logging setup
LOG_FILE = OUTPUT_DIR / "02_team_strength_ratings.log"


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


def fit_bradley_terry_margin(games_df, teams=None):
    """
    Fit Bradley-Terry model using point differential.
    Model: Margin = Strength_winner - Strength_loser + HomeAdvantage + noise

    Returns:
        strengths: dict {team_id: strength}
        home_advantage: float
    """
    from sklearn.linear_model import Ridge

    if teams is None:
        teams = list(set(games_df['WTeamID'].tolist() + games_df['LTeamID'].tolist()))

    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    log(f"  Building design matrix for {n_teams} teams, {len(games_df)} games...")

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

            if 'WLoc' in row.index:
                if row['WLoc'] == 'H':
                    X[i, -1] = 1
                elif row['WLoc'] == 'A':
                    X[i, -1] = -1

    log(f"  Fitting Ridge regression (alpha=1.0)...")
    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(X, y)

    strengths = model.coef_[:n_teams]
    strengths = strengths - strengths.mean()  # Center at 0
    home_advantage = model.coef_[-1]

    strength_dict = {teams[i]: strengths[i] for i in range(n_teams)}

    log(f"  Home court advantage: {home_advantage:.2f} points")
    log(f"  Strength range: {min(strengths):.2f} to {max(strengths):.2f}")

    return strength_dict, home_advantage


def calculate_elo_ratings(games_df, k_factor=32, home_advantage=100, initial_elo=1500):
    """
    Calculate Elo ratings from game results.
    Games must be sorted by DayNum.
    """
    elo = defaultdict(lambda: initial_elo)
    games_df = games_df.sort_values('DayNum')

    updates = 0
    for _, row in games_df.iterrows():
        winner = row['WTeamID']
        loser = row['LTeamID']

        w_elo = elo[winner]
        l_elo = elo[loser]

        # Home court adjustment
        w_elo_adj = w_elo
        l_elo_adj = l_elo

        if 'WLoc' in row.index:
            if row['WLoc'] == 'H':
                w_elo_adj = w_elo + home_advantage
            elif row['WLoc'] == 'A':
                l_elo_adj = l_elo + home_advantage

        exp_w = 1 / (1 + 10 ** ((l_elo_adj - w_elo_adj) / 400))

        elo[winner] += k_factor * (1 - exp_w)
        elo[loser] -= k_factor * (1 - exp_w)
        updates += 1

    log(f"  Processed {updates:,} Elo updates")
    log(f"  Elo range: {min(elo.values()):.0f} to {max(elo.values()):.0f}")

    return dict(elo)


def fit_glm_quality(games_sym, tourney_teams=None):
    """
    Fit GLM: PointDiff ~ Team_strength - Opponent_strength
    Coefficients ARE the team strengths (opponent-adjusted).
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        log("  WARNING: statsmodels not installed, using simplified GLM")
        return fit_glm_quality_simple(games_sym, tourney_teams)

    df = games_sym.copy()

    if tourney_teams is not None:
        mask = df['T1_TeamID'].isin(tourney_teams) | df['T2_TeamID'].isin(tourney_teams)
        df = df[mask].copy()
        log(f"  Filtered to {len(df):,} games involving tournament teams")

    df['T1_str'] = df['T1_TeamID'].astype(str)
    df['T2_str'] = df['T2_TeamID'].astype(str)

    log(f"  Fitting GLM with {df['T1_str'].nunique()} unique teams...")

    try:
        formula = "PointDiff ~ -1 + C(T1_str) + C(T2_str)"
        model = sm.GLM.from_formula(formula=formula, data=df, family=sm.families.Gaussian()).fit()

        quality = {}
        for param, value in model.params.items():
            if 'T1_str' in param:
                team_id = int(param.split('[')[1].split(']')[0].replace('T.', ''))
                quality[team_id] = value

        mean_q = np.mean(list(quality.values()))
        quality = {k: v - mean_q for k, v in quality.items()}

        log(f"  GLM Quality range: {min(quality.values()):.2f} to {max(quality.values()):.2f}")

        return quality
    except Exception as e:
        log(f"  WARNING: GLM fitting failed ({e}), using simplified method")
        return fit_glm_quality_simple(games_sym, tourney_teams)


def fit_glm_quality_simple(games_sym, tourney_teams=None):
    """
    Simplified GLM quality using average point differential adjusted for opponents.
    """
    df = games_sym.copy()

    if tourney_teams is not None:
        mask = df['T1_TeamID'].isin(tourney_teams) | df['T2_TeamID'].isin(tourney_teams)
        df = df[mask].copy()

    # First pass: average point diff for each team
    avg_diff = df.groupby('T1_TeamID')['PointDiff'].mean()

    # Second pass: adjust for opponent strength
    df['T2_strength'] = df['T2_TeamID'].map(avg_diff).fillna(0)
    df['AdjustedDiff'] = df['PointDiff'] + df['T2_strength']

    quality = df.groupby('T1_TeamID')['AdjustedDiff'].mean().to_dict()

    # Center at 0
    mean_q = np.mean(list(quality.values()))
    quality = {k: v - mean_q for k, v in quality.items()}

    log(f"  Simple GLM Quality range: {min(quality.values()):.2f} to {max(quality.values()):.2f}")

    return quality


def print_top_teams(ratings, title, n=10):
    """Print top N teams by rating."""
    log(f"\n  Top {n} teams by {title}:")
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (team, rating) in enumerate(sorted_ratings[:n], 1):
        log(f"    {i:2d}. Team {team}: {rating:.2f}")


def main():
    """Main execution function."""
    # Clear previous log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    print_separator("PHASE 2: TEAM STRENGTH RATINGS")

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print_separator("Loading Processed Data")

    regular_sym_path = PROCESSED_DIR / "regular_sym.csv"
    seeds_path = PROCESSED_DIR / "seeds.csv"
    regular_results_path = PROCESSED_DIR / "regular_results.csv"

    if not regular_sym_path.exists():
        log(f"ERROR: {regular_sym_path} not found. Run 01_data_preparation.py first.")
        sys.exit(1)

    regular_sym = pd.read_csv(regular_sym_path)
    log(f"Loaded regular_sym: {len(regular_sym):,} records")

    if regular_results_path.exists():
        regular_results = pd.read_csv(regular_results_path)
        log(f"Loaded regular_results: {len(regular_results):,} games")
    else:
        log("WARNING: regular_results.csv not found, will reconstruct from symmetric")
        regular_results = None

    if seeds_path.exists():
        seeds = pd.read_csv(seeds_path)
        log(f"Loaded seeds: {len(seeds):,} entries")
    else:
        log("WARNING: seeds.csv not found")
        seeds = pd.DataFrame()

    # Get list of seasons
    seasons = sorted(regular_sym['Season'].unique())
    log(f"\nSeasons to process: {seasons}")

    # Determine genders in data
    if 'Gender' in regular_sym.columns:
        genders = sorted(regular_sym['Gender'].unique())
    else:
        # Infer from team IDs: Men's < 2000, Women's >= 3000
        regular_sym['Gender'] = regular_sym['T1_TeamID'].apply(lambda x: 'M' if x < 2000 else 'W')
        if regular_results is not None and 'Gender' not in regular_results.columns:
            regular_results['Gender'] = regular_results['WTeamID'].apply(lambda x: 'M' if x < 2000 else 'W')
        genders = ['M', 'W']

    log(f"Genders in data: {genders}")
    log("NOTE: Men's and Women's ratings computed SEPARATELY")

    # =========================================================================
    # COMPUTE BRADLEY-TERRY RATINGS
    # =========================================================================
    print_separator("Computing Bradley-Terry Ratings (per season, per gender)")

    bt_ratings = {}
    bt_home_advantages = {}

    for season in seasons:
        bt_ratings[season] = {}
        bt_home_advantages[season] = {}

        for gender in genders:
            log(f"\nSeason {season} - {gender}:")

            if regular_results is not None:
                season_games = regular_results[
                    (regular_results['Season'] == season) &
                    (regular_results['Gender'] == gender)
                ]
            else:
                # Reconstruct from symmetric (take winner perspective only)
                season_sym = regular_sym[
                    (regular_sym['Season'] == season) &
                    (regular_sym['Gender'] == gender)
                ]
                season_games = season_sym[season_sym['T1_Win'] == 1].copy()
                season_games = season_games.rename(columns={
                    'T1_TeamID': 'WTeamID',
                    'T2_TeamID': 'LTeamID',
                    'T1_Score': 'WScore',
                    'T2_Score': 'LScore'
                })

            if len(season_games) == 0:
                log(f"  No games found, skipping")
                continue

            log(f"  Games: {len(season_games):,}")

            strengths, home_adv = fit_bradley_terry_margin(season_games)
            bt_ratings[season].update(strengths)  # Merge into season dict
            bt_home_advantages[season][gender] = home_adv

            print_top_teams(strengths, f"Bradley-Terry strength ({gender})", n=5)

    # =========================================================================
    # COMPUTE ELO RATINGS
    # =========================================================================
    print_separator("Computing Elo Ratings (per season, per gender)")

    elo_ratings = {}

    for season in seasons:
        elo_ratings[season] = {}

        for gender in genders:
            log(f"\nSeason {season} - {gender}:")

            if regular_results is not None:
                season_games = regular_results[
                    (regular_results['Season'] == season) &
                    (regular_results['Gender'] == gender)
                ].copy()
            else:
                season_sym = regular_sym[
                    (regular_sym['Season'] == season) &
                    (regular_sym['Gender'] == gender)
                ]
                season_games = season_sym[season_sym['T1_Win'] == 1].copy()
                season_games = season_games.rename(columns={
                    'T1_TeamID': 'WTeamID',
                    'T2_TeamID': 'LTeamID',
                    'T1_Score': 'WScore',
                    'T2_Score': 'LScore'
                })

            if len(season_games) == 0:
                log(f"  No games found, skipping")
                continue

            log(f"  Games: {len(season_games):,}")

            elo = calculate_elo_ratings(season_games)
            elo_ratings[season].update(elo)  # Merge into season dict

            print_top_teams(elo, f"Elo ({gender})", n=5)

    # =========================================================================
    # COMPUTE GLM QUALITY RATINGS
    # =========================================================================
    print_separator("Computing GLM Quality Ratings (per season, per gender)")

    glm_quality = {}

    for season in seasons:
        glm_quality[season] = {}

        for gender in genders:
            log(f"\nSeason {season} - {gender}:")

            season_sym = regular_sym[
                (regular_sym['Season'] == season) &
                (regular_sym['Gender'] == gender)
            ]

            if len(season_sym) == 0:
                log(f"  No games found, skipping")
                continue

            log(f"  Symmetric games: {len(season_sym):,}")

            # Get tournament teams for this season/gender (if available)
            tourney_teams = None
            if len(seeds) > 0 and 'Season' in seeds.columns:
                season_seeds = seeds[
                    (seeds['Season'] == season) &
                    (seeds['Gender'] == gender)
                ]
                if len(season_seeds) > 0:
                    tourney_teams = set(season_seeds['TeamID'].unique())
                    log(f"  Tournament teams: {len(tourney_teams)}")

            quality = fit_glm_quality(season_sym, tourney_teams)
            glm_quality[season].update(quality)  # Merge into season dict

            print_top_teams(quality, f"GLM Quality ({gender})", n=5)

    # =========================================================================
    # SAVE RATINGS
    # =========================================================================
    print_separator("Saving Ratings")

    # Save as pickle for easy loading
    with open(PROCESSED_DIR / "bradley_terry_ratings.pkl", "wb") as f:
        pickle.dump({'ratings': bt_ratings, 'home_advantages': bt_home_advantages}, f)
    log(f"Saved: {PROCESSED_DIR / 'bradley_terry_ratings.pkl'}")

    with open(PROCESSED_DIR / "elo_ratings.pkl", "wb") as f:
        pickle.dump(elo_ratings, f)
    log(f"Saved: {PROCESSED_DIR / 'elo_ratings.pkl'}")

    with open(PROCESSED_DIR / "glm_quality_ratings.pkl", "wb") as f:
        pickle.dump(glm_quality, f)
    log(f"Saved: {PROCESSED_DIR / 'glm_quality_ratings.pkl'}")

    # Also save as CSV for inspection
    all_ratings_rows = []

    for season in seasons:
        teams = set(bt_ratings[season].keys()) | set(elo_ratings[season].keys()) | set(glm_quality[season].keys())

        for team in teams:
            all_ratings_rows.append({
                'Season': season,
                'TeamID': team,
                'BT_strength': bt_ratings[season].get(team, np.nan),
                'Elo': elo_ratings[season].get(team, 1500),
                'GLM_quality': glm_quality[season].get(team, np.nan)
            })

    all_ratings_df = pd.DataFrame(all_ratings_rows)
    all_ratings_df.to_csv(PROCESSED_DIR / "all_ratings.csv", index=False)
    log(f"Saved: {PROCESSED_DIR / 'all_ratings.csv'}")

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print_separator("PHASE 2 COMPLETE - SUMMARY")

    log(f"\nRatings computed for {len(seasons)} seasons")

    # Overall statistics
    all_bt = [v for season_dict in bt_ratings.values() for v in season_dict.values()]
    all_elo = [v for season_dict in elo_ratings.values() for v in season_dict.values()]
    all_glm = [v for season_dict in glm_quality.values() for v in season_dict.values()]

    log(f"\nBradley-Terry strength:")
    log(f"  Mean: {np.mean(all_bt):.2f}")
    log(f"  Std: {np.std(all_bt):.2f}")
    log(f"  Range: {min(all_bt):.2f} to {max(all_bt):.2f}")

    log(f"\nElo ratings:")
    log(f"  Mean: {np.mean(all_elo):.0f}")
    log(f"  Std: {np.std(all_elo):.0f}")
    log(f"  Range: {min(all_elo):.0f} to {max(all_elo):.0f}")

    log(f"\nGLM Quality:")
    log(f"  Mean: {np.mean(all_glm):.2f}")
    log(f"  Std: {np.std(all_glm):.2f}")
    log(f"  Range: {min(all_glm):.2f} to {max(all_glm):.2f}")

    log(f"\nHome court advantage (Bradley-Terry):")
    # Flatten nested dict {season: {gender: value}} to list of values
    all_home_adv = [v for season_dict in bt_home_advantages.values() for v in season_dict.values()]
    if all_home_adv:
        log(f"  Mean: {np.mean(all_home_adv):.2f} points")
        log(f"  Range: {min(all_home_adv):.2f} to {max(all_home_adv):.2f}")

    # Correlation between ratings
    log(f"\nRating correlations (from combined CSV):")
    corr_matrix = all_ratings_df[['BT_strength', 'Elo', 'GLM_quality']].corr()
    log(f"  BT vs Elo: {corr_matrix.loc['BT_strength', 'Elo']:.3f}")
    log(f"  BT vs GLM: {corr_matrix.loc['BT_strength', 'GLM_quality']:.3f}")
    log(f"  Elo vs GLM: {corr_matrix.loc['Elo', 'GLM_quality']:.3f}")

    log(f"\nOutputs saved to: {PROCESSED_DIR}")
    log(f"Log saved to: {LOG_FILE}")

    print_separator()

    return 0


if __name__ == "__main__":
    sys.exit(main())
