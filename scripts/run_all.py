#!/usr/bin/env python3
"""
Master Pipeline Runner
======================
Runs all phases of the March Machine Learning Mania pipeline in sequence.

Phases:
1. Data Preparation
2. Team Strength Ratings
3. Feature Engineering
4. Model Training
5. Calibration & Submission

Usage:
    python run_all.py              # Run all phases
    python run_all.py --phase 3    # Start from phase 3
    python run_all.py --phase 1-3  # Run phases 1 through 3

Outputs are saved to the 'outputs' folder with detailed logs for each phase.
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)

# Master log file
MASTER_LOG = OUTPUT_DIR / "00_master_pipeline.log"

# Phase scripts
PHASES = [
    ("01_data_preparation.py", "Phase 1: Data Preparation"),
    ("02_team_strength_ratings.py", "Phase 2: Team Strength Ratings"),
    ("03_feature_engineering.py", "Phase 3: Feature Engineering"),
    ("04_model_training.py", "Phase 4: Model Training"),
    ("05_calibration_submission.py", "Phase 5: Calibration & Submission"),
]


def log(message, also_print=True):
    """Log message to master log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"

    with open(MASTER_LOG, "a") as f:
        f.write(log_message + "\n")

    if also_print:
        print(log_message)


def print_banner():
    """Print pipeline banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          MARCH MACHINE LEARNING MANIA 2026 - PIPELINE RUNNER                 ║
║                                                                              ║
║   Based on analysis of winning solutions from recent competitions            ║
║   Using: Bradley-Terry + Elo + GLM Quality + XGBoost                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    log(banner, also_print=False)


def run_phase(script_name, phase_title, phase_num):
    """Run a single phase script."""
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        log(f"ERROR: Script not found: {script_path}")
        return False

    log(f"\n{'='*70}")
    log(f"STARTING: {phase_title}")
    log(f"Script: {script_name}")
    log(f"{'='*70}")

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_DIR),
            capture_output=False,  # Let output go to terminal
            check=True
        )

        elapsed = time.time() - start_time
        log(f"\n{'='*70}")
        log(f"COMPLETED: {phase_title}")
        log(f"Elapsed time: {elapsed:.1f} seconds")
        log(f"{'='*70}")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        log(f"\n{'='*70}")
        log(f"FAILED: {phase_title}")
        log(f"Exit code: {e.returncode}")
        log(f"Elapsed time: {elapsed:.1f} seconds")
        log(f"{'='*70}")
        return False

    except Exception as e:
        log(f"ERROR running {script_name}: {e}")
        return False


def parse_phase_range(phase_arg):
    """Parse phase argument (e.g., '3', '1-3', '2-5')."""
    if phase_arg is None:
        return 1, 5

    if '-' in phase_arg:
        parts = phase_arg.split('-')
        start = int(parts[0])
        end = int(parts[1])
    else:
        start = int(phase_arg)
        end = 5  # Run from start to end

    return max(1, start), min(5, end)


def check_prerequisites(start_phase):
    """Check if prerequisites for starting phase exist."""
    if start_phase == 1:
        # Check for data directory
        data_dir = PROJECT_DIR / "data"
        if not data_dir.exists():
            log("\n" + "="*70)
            log("SETUP REQUIRED")
            log("="*70)
            log(f"\nData directory not found: {data_dir}")
            log("\nPlease create the 'data' folder and add Kaggle competition files:")
            log("  - MRegularSeasonDetailedResults.csv")
            log("  - WRegularSeasonDetailedResults.csv")
            log("  - MNCAATourneyDetailedResults.csv")
            log("  - WNCAATourneyDetailedResults.csv")
            log("  - MNCAATourneySeeds.csv")
            log("  - WNCAATourneySeeds.csv")
            log("\nDownload from: https://www.kaggle.com/c/march-machine-learning-mania-2026/data")
            return False
    else:
        # Check for processed files from previous phases
        processed_dir = PROJECT_DIR / "processed"
        required_files = {
            2: ["regular_sym.csv"],
            3: ["regular_sym.csv", "bradley_terry_ratings.pkl", "elo_ratings.pkl"],
            4: ["training_features.csv"],
            5: ["xgb_loso_models.pkl"],
        }

        if start_phase in required_files:
            for filename in required_files[start_phase]:
                # Check in processed dir or models dir
                if "model" in filename or filename.endswith(".pkl"):
                    filepath = PROJECT_DIR / "models" / filename
                    if not filepath.exists():
                        filepath = processed_dir / filename
                else:
                    filepath = processed_dir / filename

                if not filepath.exists():
                    log(f"\nERROR: Required file not found: {filepath}")
                    log(f"Please run earlier phases first (starting from phase 1).")
                    return False

    return True


def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(description="Run March Machine Learning Mania pipeline")
    parser.add_argument("--phase", type=str, default=None,
                        help="Phase to start from (e.g., '3' or '1-3')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run without executing")
    args = parser.parse_args()

    # Clear master log for new run
    if MASTER_LOG.exists() and args.phase is None:
        MASTER_LOG.unlink()

    print_banner()

    log(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Project directory: {PROJECT_DIR}")
    log(f"Output directory: {OUTPUT_DIR}")

    # Parse phase range
    start_phase, end_phase = parse_phase_range(args.phase)
    log(f"Running phases {start_phase} through {end_phase}")

    # Check prerequisites
    if not args.dry_run:
        if not check_prerequisites(start_phase):
            return 1

    # Run phases
    phases_to_run = PHASES[start_phase-1:end_phase]
    total_phases = len(phases_to_run)
    successful = 0
    failed = 0

    pipeline_start = time.time()

    for i, (script, title) in enumerate(phases_to_run, start_phase):
        log(f"\n[{i}/{start_phase + total_phases - 1}] {title}")

        if args.dry_run:
            log(f"  Would run: {script}")
            successful += 1
            continue

        if run_phase(script, title, i):
            successful += 1
        else:
            failed += 1
            log(f"\nPipeline stopped due to failure in {title}")
            break

    pipeline_elapsed = time.time() - pipeline_start

    # Summary
    log("\n" + "="*70)
    log("PIPELINE SUMMARY")
    log("="*70)
    log(f"\nPhases run: {successful + failed}")
    log(f"Successful: {successful}")
    log(f"Failed: {failed}")
    log(f"Total time: {pipeline_elapsed:.1f} seconds ({pipeline_elapsed/60:.1f} minutes)")

    if failed == 0 and not args.dry_run:
        log("\n" + "="*70)
        log("SUCCESS! All phases completed.")
        log("="*70)
        log(f"\nOutputs:")
        log(f"  - Logs: {OUTPUT_DIR}")
        log(f"  - Processed data: {PROJECT_DIR / 'processed'}")
        log(f"  - Models: {PROJECT_DIR / 'models'}")
        log(f"  - Submissions: {PROJECT_DIR / 'submissions'}")
        log(f"\nSubmission files ready for Kaggle!")
    else:
        log("\nPipeline completed with errors. Check individual phase logs for details.")

    log(f"\nMaster log saved to: {MASTER_LOG}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
