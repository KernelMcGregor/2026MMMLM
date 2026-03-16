#!/usr/bin/env python3
"""
Phase 1: Data Preparation
=========================
Load Kaggle data, clean, merge men's/women's, normalize overtime.

Outputs:
- regular_results.csv: Combined regular season data
- tourney_results.csv: Combined tournament data
- seeds.csv: Parsed tournament seeds
- regular_sym.csv: Symmetric game dataframe
- tourney_sym.csv: Symmetric tournament games
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PROCESSED_DIR = PROJECT_DIR / "processed"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Logging setup
LOG_FILE = OUTPUT_DIR / "01_data_preparation.log"


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


def print_df_info(df, name):
    """Print detailed dataframe info."""
    log(f"\n--- {name} ---")
    log(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    log(f"  Columns: {list(df.columns)}")
    log(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    if 'Season' in df.columns:
        log(f"  Seasons: {sorted(df['Season'].unique())}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        missing_cols = missing[missing > 0]
        log(f"  Missing values: {dict(missing_cols)}")
    else:
        log(f"  Missing values: None")


def normalize_overtime(df):
    """Normalize stats to 40-minute game equivalent."""
    df = df.copy()

    if 'NumOT' not in df.columns:
        log("  No NumOT column found, skipping normalization")
        return df

    ot_games = (df['NumOT'] > 0).sum()
    log(f"  Found {ot_games:,} overtime games ({ot_games/len(df)*100:.1f}%)")

    ot_factor = 40 / (40 + 5 * df['NumOT'])

    stat_cols = ['WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
                 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
                 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

    normalized_cols = 0
    for col in stat_cols:
        if col in df.columns:
            df[col] = df[col] * ot_factor
            normalized_cols += 1

    log(f"  Normalized {normalized_cols} stat columns")
    return df


def create_symmetric_games(df):
    """Each game appears from both team perspectives."""
    log("  Creating symmetric game dataframe...")

    # Preserve Gender column if present
    has_gender = 'Gender' in df.columns

    # Winner perspective
    df1 = df.copy()
    df1['T1_TeamID'] = df['WTeamID']
    df1['T2_TeamID'] = df['LTeamID']
    df1['T1_Score'] = df['WScore']
    df1['T2_Score'] = df['LScore']
    df1['PointDiff'] = df1['T1_Score'] - df1['T2_Score']
    df1['T1_Win'] = 1

    if 'WHome' in df.columns:
        df1['T1_Home'] = df['WHome']
        df1['T2_Home'] = -df['WHome']
    else:
        df1['T1_Home'] = np.nan
        df1['T2_Home'] = np.nan

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

    if 'WHome' in df.columns:
        df2['T1_Home'] = -df['WHome']
        df2['T2_Home'] = df['WHome']
    else:
        df2['T1_Home'] = np.nan
        df2['T2_Home'] = np.nan

    for prefix in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
        if f'W{prefix}' in df.columns:
            df2[f'T1_{prefix}'] = df[f'L{prefix}']
            df2[f'T2_{prefix}'] = df[f'W{prefix}']

    result = pd.concat([df1, df2], ignore_index=True)

    # Add Gender if not present (infer from team ID)
    if not has_gender and 'T1_TeamID' in result.columns:
        result['Gender'] = result['T1_TeamID'].apply(lambda x: 'M' if x < 2000 else 'W')

    log(f"  Created {len(result):,} symmetric game records from {len(df):,} games")

    if has_gender or 'Gender' in result.columns:
        gender_counts = result['Gender'].value_counts()
        for g, c in gender_counts.items():
            log(f"    {g}: {c:,} records")

    return result


def parse_seed(seed_str):
    """Extract numeric seed from seed string (e.g., 'W01' -> 1)."""
    return int(seed_str[1:3])


def main():
    """Main execution function."""
    # Clear previous log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    print_separator("PHASE 1: DATA PREPARATION")
    log(f"Project directory: {PROJECT_DIR}")
    log(f"Data directory: {DATA_DIR}")
    log(f"Output directory: {OUTPUT_DIR}")

    # Check if data directory exists
    if not DATA_DIR.exists():
        log(f"\nERROR: Data directory not found at {DATA_DIR}")
        log("Please create a 'data' folder with Kaggle competition data files:")
        log("  - MRegularSeasonDetailedResults.csv")
        log("  - WRegularSeasonDetailedResults.csv")
        log("  - MNCAATourneyDetailedResults.csv")
        log("  - WNCAATourneyDetailedResults.csv")
        log("  - MNCAATourneySeeds.csv")
        log("  - WNCAATourneySeeds.csv")
        sys.exit(1)

    # =========================================================================
    # STEP 1.1: Load and Combine Data
    # =========================================================================
    print_separator("STEP 1.1: Loading Data Files")

    # Men's data
    m_regular_path = DATA_DIR / "MRegularSeasonDetailedResults.csv"
    w_regular_path = DATA_DIR / "WRegularSeasonDetailedResults.csv"
    m_tourney_path = DATA_DIR / "MNCAATourneyDetailedResults.csv"
    w_tourney_path = DATA_DIR / "WNCAATourneyDetailedResults.csv"
    m_seeds_path = DATA_DIR / "MNCAATourneySeeds.csv"
    w_seeds_path = DATA_DIR / "WNCAATourneySeeds.csv"

    # Load men's regular season
    if m_regular_path.exists():
        m_regular = pd.read_csv(m_regular_path)
        log(f"Loaded MRegularSeasonDetailedResults: {len(m_regular):,} games")
    else:
        log(f"WARNING: {m_regular_path} not found")
        m_regular = pd.DataFrame()

    # Load women's regular season
    if w_regular_path.exists():
        w_regular = pd.read_csv(w_regular_path)
        log(f"Loaded WRegularSeasonDetailedResults: {len(w_regular):,} games")
    else:
        log(f"WARNING: {w_regular_path} not found")
        w_regular = pd.DataFrame()

    # Load men's tournament
    if m_tourney_path.exists():
        m_tourney = pd.read_csv(m_tourney_path)
        log(f"Loaded MNCAATourneyDetailedResults: {len(m_tourney):,} games")
    else:
        log(f"WARNING: {m_tourney_path} not found")
        m_tourney = pd.DataFrame()

    # Load women's tournament
    if w_tourney_path.exists():
        w_tourney = pd.read_csv(w_tourney_path)
        log(f"Loaded WNCAATourneyDetailedResults: {len(w_tourney):,} games")
    else:
        log(f"WARNING: {w_tourney_path} not found")
        w_tourney = pd.DataFrame()

    # Load seeds
    if m_seeds_path.exists():
        m_seeds = pd.read_csv(m_seeds_path)
        log(f"Loaded MNCAATourneySeeds: {len(m_seeds):,} entries")
    else:
        log(f"WARNING: {m_seeds_path} not found")
        m_seeds = pd.DataFrame()

    if w_seeds_path.exists():
        w_seeds = pd.read_csv(w_seeds_path)
        log(f"Loaded WNCAATourneySeeds: {len(w_seeds):,} entries")
    else:
        log(f"WARNING: {w_seeds_path} not found")
        w_seeds = pd.DataFrame()

    # Keep men's and women's data SEPARATE (they are independent competitions)
    print_separator("STEP 1.2: Processing Men's and Women's Data Separately")

    log("NOTE: Men's and Women's data are kept SEPARATE throughout the pipeline.")
    log("They are independent competitions with no crossover.")

    # Add gender indicator
    if len(m_regular) > 0:
        m_regular['Gender'] = 'M'
    if len(w_regular) > 0:
        w_regular['Gender'] = 'W'
    if len(m_tourney) > 0:
        m_tourney['Gender'] = 'M'
    if len(w_tourney) > 0:
        w_tourney['Gender'] = 'W'
    if len(m_seeds) > 0:
        m_seeds['Gender'] = 'M'
    if len(w_seeds) > 0:
        w_seeds['Gender'] = 'W'

    # Combine but with Gender column for filtering
    regular_results = pd.concat([m_regular, w_regular], ignore_index=True)
    tourney_results = pd.concat([m_tourney, w_tourney], ignore_index=True)
    seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)

    log(f"\nMen's regular season: {len(m_regular):,} games")
    log(f"Women's regular season: {len(w_regular):,} games")
    log(f"Men's tournament: {len(m_tourney):,} games")
    log(f"Women's tournament: {len(w_tourney):,} games")
    log(f"Men's seeds: {len(m_seeds):,} entries")
    log(f"Women's seeds: {len(w_seeds):,} entries")

    print_df_info(regular_results, "All Regular Season (with Gender column)")
    print_df_info(tourney_results, "All Tournament (with Gender column)")
    print_df_info(seeds, "All Seeds (with Gender column)")

    # =========================================================================
    # STEP 1.2: Normalize Overtime & Add Home Court
    # =========================================================================
    print_separator("STEP 1.3: Normalizing Overtime Stats")

    log("\nRegular season normalization:")
    regular_results = normalize_overtime(regular_results)

    log("\nTournament normalization:")
    tourney_results = normalize_overtime(tourney_results)

    # Add home court indicator
    print_separator("STEP 1.4: Adding Home Court Indicator")

    wloc = {'H': 1, 'A': -1, 'N': 0}

    if 'WLoc' in regular_results.columns:
        regular_results['WHome'] = regular_results['WLoc'].map(lambda x: wloc.get(x, 0))
        home_counts = regular_results['WLoc'].value_counts()
        log(f"Home court distribution in regular season:")
        for loc, count in home_counts.items():
            log(f"  {loc}: {count:,} games ({count/len(regular_results)*100:.1f}%)")
    else:
        regular_results['WHome'] = 0
        log("No WLoc column found, setting WHome to 0 (neutral)")

    # Tournament games are neutral
    tourney_results['WHome'] = 0
    log("Tournament games set to neutral (WHome=0)")

    # =========================================================================
    # STEP 1.3: Create Symmetric Game DataFrames
    # =========================================================================
    print_separator("STEP 1.5: Creating Symmetric Game DataFrames")

    log("\nRegular season:")
    regular_sym = create_symmetric_games(regular_results)
    print_df_info(regular_sym, "Symmetric Regular Season")

    log("\nTournament:")
    tourney_sym = create_symmetric_games(tourney_results)
    print_df_info(tourney_sym, "Symmetric Tournament")

    # Point differential analysis
    log("\n--- Point Differential Analysis ---")
    log(f"  Regular season mean: {regular_sym['PointDiff'].mean():.2f}")
    log(f"  Regular season std: {regular_sym['PointDiff'].std():.2f}")
    log(f"  Regular season min/max: {regular_sym['PointDiff'].min():.0f} / {regular_sym['PointDiff'].max():.0f}")
    log(f"  Tournament mean: {tourney_sym['PointDiff'].mean():.2f}")
    log(f"  Tournament std: {tourney_sym['PointDiff'].std():.2f}")

    # =========================================================================
    # STEP 1.4: Parse Seeds
    # =========================================================================
    print_separator("STEP 1.6: Parsing Tournament Seeds")

    if len(seeds) > 0:
        seeds['SeedNum'] = seeds['Seed'].apply(parse_seed)

        log(f"\nSeed distribution:")
        seed_counts = seeds['SeedNum'].value_counts().sort_index()
        for seed, count in seed_counts.items():
            log(f"  Seed {seed:2d}: {count:,} teams")

        # Region distribution
        if 'Seed' in seeds.columns:
            seeds['Region'] = seeds['Seed'].str[0]
            region_counts = seeds['Region'].value_counts().sort_index()
            log(f"\nRegion distribution:")
            for region, count in region_counts.items():
                log(f"  {region}: {count:,} teams")

    print_df_info(seeds, "Parsed Seeds")

    # =========================================================================
    # SAVE PROCESSED DATA
    # =========================================================================
    print_separator("SAVING PROCESSED DATA")

    # Save to processed directory
    regular_results.to_csv(PROCESSED_DIR / "regular_results.csv", index=False)
    log(f"Saved: {PROCESSED_DIR / 'regular_results.csv'}")

    tourney_results.to_csv(PROCESSED_DIR / "tourney_results.csv", index=False)
    log(f"Saved: {PROCESSED_DIR / 'tourney_results.csv'}")

    seeds.to_csv(PROCESSED_DIR / "seeds.csv", index=False)
    log(f"Saved: {PROCESSED_DIR / 'seeds.csv'}")

    regular_sym.to_csv(PROCESSED_DIR / "regular_sym.csv", index=False)
    log(f"Saved: {PROCESSED_DIR / 'regular_sym.csv'}")

    tourney_sym.to_csv(PROCESSED_DIR / "tourney_sym.csv", index=False)
    log(f"Saved: {PROCESSED_DIR / 'tourney_sym.csv'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_separator("PHASE 1 COMPLETE - SUMMARY")

    log(f"\nData loaded and processed:")
    log(f"  - Regular season games: {len(regular_results):,}")
    log(f"  - Tournament games: {len(tourney_results):,}")
    log(f"  - Seeds entries: {len(seeds):,}")
    log(f"  - Symmetric regular records: {len(regular_sym):,}")
    log(f"  - Symmetric tournament records: {len(tourney_sym):,}")

    if 'Season' in regular_results.columns:
        seasons = sorted(regular_results['Season'].unique())
        log(f"  - Seasons covered: {min(seasons)} to {max(seasons)} ({len(seasons)} seasons)")

    unique_teams = len(set(regular_sym['T1_TeamID'].unique()))
    log(f"  - Unique teams: {unique_teams:,}")

    log(f"\nOutputs saved to: {PROCESSED_DIR}")
    log(f"Log saved to: {LOG_FILE}")

    print_separator()

    return 0


if __name__ == "__main__":
    sys.exit(main())
