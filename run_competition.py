#!/usr/bin/env python3
"""
Script to run the multi-agent system on MLE-bench competitions.

This script helps you:
1. List available competitions (with filtering by complexity)
2. Prepare a competition (downloads and splits data)
3. Run the multi-agent system on the competition
4. Grade the submission

Usage:
    # List all competitions
    python run_competition.py --list

    # List only low complexity (lite) competitions
    python run_competition.py --list --lite

    # Run on a specific competition
    python run_competition.py --competition tabular-playground-series-dec-2021

    # Run with custom time budget (default: 300s = 5 minutes)
    python run_competition.py --competition nomad2018-predict-transparent-conductors --time-budget 600
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add mle_bench_agents to path
sys.path.insert(0, str(Path(__file__).parent))

from mle_bench_agents.runner import CompetitionRunner


def get_available_competitions() -> List[str]:
    """Get list of all available competitions."""
    competitions_dir = Path(__file__).parent / "mlebench" / "competitions"
    competitions = []

    for comp_dir in competitions_dir.iterdir():
        if comp_dir.is_dir() and not comp_dir.name.startswith("__"):
            config_file = comp_dir / "config.yaml"
            if config_file.exists():
                competitions.append(comp_dir.name)

    return sorted(competitions)


def get_lite_competitions() -> List[str]:
    """Get list of low complexity (lite) competitions."""
    lite_file = Path(__file__).parent / "experiments" / "splits" / "low.txt"

    if not lite_file.exists():
        print(f"Warning: Lite competitions file not found at {lite_file}")
        return []

    with open(lite_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def list_competitions(lite_only: bool = False):
    """List available competitions."""
    if lite_only:
        competitions = get_lite_competitions()
        print("\n" + "=" * 80)
        print("LOW COMPLEXITY (LITE) COMPETITIONS (22 total)")
        print("=" * 80)
        print("\nThese are faster and use less data, perfect for testing!\n")
    else:
        competitions = get_available_competitions()
        print("\n" + "=" * 80)
        print(f"ALL AVAILABLE COMPETITIONS ({len(competitions)} total)")
        print("=" * 80)
        print()

    # Group by category
    tabular = []
    text = []
    image = []
    audio = []
    other = []

    for comp in competitions:
        comp_lower = comp.lower()
        if any(x in comp_lower for x in ["tabular", "taxi", "nomad", "playground"]):
            tabular.append(comp)
        elif any(x in comp_lower for x in ["text", "comment", "toxic", "insults", "pizza", "spooky", "author"]):
            text.append(comp)
        elif any(x in comp_lower for x in ["image", "cactus", "dog", "cat", "leaf", "cancer", "blindness", "document", "melanoma", "pathology"]):
            image.append(comp)
        elif any(x in comp_lower for x in ["audio", "bird", "whale", "speech"]):
            audio.append(comp)
        else:
            other.append(comp)

    if tabular:
        print("TABULAR COMPETITIONS:")
        for comp in tabular:
            print(f"  - {comp}")
        print()

    if text:
        print("TEXT COMPETITIONS:")
        for comp in text:
            print(f"  - {comp}")
        print()

    if image:
        print("IMAGE COMPETITIONS:")
        for comp in image:
            print(f"  - {comp}")
        print()

    if audio:
        print("AUDIO COMPETITIONS:")
        for comp in audio:
            print(f"  - {comp}")
        print()

    if other:
        print("OTHER COMPETITIONS:")
        for comp in other:
            print(f"  - {comp}")
        print()

    print("=" * 80)

    if not lite_only:
        print("\nTip: Use --lite flag to see only low complexity competitions")
        print("     These are recommended for initial testing!")

    print("\nTo run a competition, use:")
    print("  python run_competition.py --competition <competition-id>")
    print()


def prepare_competition(competition_id: str) -> Optional[Path]:
    """
    Prepare a competition by downloading and splitting data.

    Returns the path to the prepared public data directory, or None if preparation failed.
    """
    print("\n" + "=" * 80)
    print(f"PREPARING COMPETITION: {competition_id}")
    print("=" * 80)
    print()

    try:
        # Run mlebench prepare command
        cmd = ["mlebench", "prepare", "-c", competition_id]
        print(f"Running: {' '.join(cmd)}")
        print()

        result = subprocess.run(cmd, check=True, capture_output=False, text=True)

        # Find the prepared data directory
        import platformdirs
        cache_dir = Path(platformdirs.user_cache_dir("mlebench"))
        public_dir = cache_dir / competition_id / "prepared" / "public"

        if public_dir.exists():
            print(f"\n✓ Competition prepared successfully!")
            print(f"✓ Public data directory: {public_dir}")
            return public_dir
        else:
            print(f"\n✗ Warning: Expected data directory not found at {public_dir}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error preparing competition: {e}")
        return None
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return None


def run_multi_agent_system(competition_id: str, data_path: Path, time_budget: int = 300) -> Optional[Path]:
    """
    Run the multi-agent system on a competition.

    Returns the path to the submission file, or None if run failed.
    """
    print("\n" + "=" * 80)
    print(f"RUNNING MULTI-AGENT SYSTEM")
    print("=" * 80)
    print(f"Competition: {competition_id}")
    print(f"Data Path: {data_path}")
    print(f"Time Budget: {time_budget}s ({time_budget/60:.1f} minutes)")
    print("=" * 80)
    print()

    try:
        # Create output directory for submission
        submission_dir = Path(f"/tmp/{competition_id}_submission")
        submission_dir.mkdir(exist_ok=True, parents=True)

        # Create logs directory
        log_dir = Path(f"/tmp/{competition_id}_logs")
        log_dir.mkdir(exist_ok=True, parents=True)

        # Initialize and run the competition runner
        runner = CompetitionRunner(log_dir=log_dir, log_level="INFO")

        result = runner.run_competition(
            competition_id=competition_id,
            data_path=data_path,
            submission_path=submission_dir,
            time_budget=time_budget
        )

        print("\n" + "=" * 80)
        print("MULTI-AGENT SYSTEM RESULTS")
        print("=" * 80)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Submission Path: {result.get('submission_path', 'N/A')}")
        print(f"Best CV Score: {result.get('best_cv_score', 'N/A')}")
        print(f"Models Trained: {result.get('models_trained', 'N/A')}")
        print(f"Ensemble Used: {result.get('ensemble_used', False)}")
        print("=" * 80)
        print()

        submission_path = result.get("submission_path")
        if submission_path and Path(submission_path).exists():
            return Path(submission_path)
        else:
            print("✗ Warning: Submission file not created")
            return None

    except Exception as e:
        print(f"\n✗ Error running multi-agent system: {e}")
        import traceback
        traceback.print_exc()
        return None


def grade_submission(competition_id: str, submission_path: Path):
    """Grade a submission using mlebench."""
    print("\n" + "=" * 80)
    print(f"GRADING SUBMISSION")
    print("=" * 80)
    print()

    try:
        # Run mlebench grade-sample command
        cmd = ["mlebench", "grade-sample", str(submission_path), competition_id]
        print(f"Running: {' '.join(cmd)}")
        print()

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print("\n✓ Grading completed successfully!")
        else:
            print(f"\n✗ Grading failed with return code {result.returncode}")

    except Exception as e:
        print(f"\n✗ Error grading submission: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent system on MLE-bench competitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all competitions
  python run_competition.py --list

  # List only low complexity competitions
  python run_competition.py --list --lite

  # Run on a tabular competition (recommended for first test)
  python run_competition.py --competition tabular-playground-series-dec-2021

  # Run on a small tabular competition
  python run_competition.py --competition nomad2018-predict-transparent-conductors

  # Run with custom time budget
  python run_competition.py --competition tabular-playground-series-dec-2021 --time-budget 600

  # Skip preparation if already done
  python run_competition.py --competition tabular-playground-series-dec-2021 --skip-prepare
        """
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available competitions and exit"
    )

    parser.add_argument(
        "--lite",
        action="store_true",
        help="Show only low complexity (lite) competitions (use with --list)"
    )

    parser.add_argument(
        "--competition", "-c",
        type=str,
        help="Competition ID to run"
    )

    parser.add_argument(
        "--time-budget", "-t",
        type=int,
        default=300,
        help="Time budget in seconds (default: 300 = 5 minutes)"
    )

    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip preparation step (assumes competition is already prepared)"
    )

    parser.add_argument(
        "--skip-grading",
        action="store_true",
        help="Skip grading step (just run the agent)"
    )

    args = parser.parse_args()

    # List competitions
    if args.list:
        list_competitions(lite_only=args.lite)
        return

    # Run competition
    if not args.competition:
        parser.print_help()
        print("\nError: --competition is required (or use --list to see available competitions)")
        sys.exit(1)

    competition_id = args.competition

    # Step 1: Prepare competition (unless skipped)
    if args.skip_prepare:
        print("\nSkipping preparation step...")
        import platformdirs
        cache_dir = Path(platformdirs.user_cache_dir("mlebench"))
        data_path = cache_dir / competition_id / "prepared" / "public"

        if not data_path.exists():
            print(f"✗ Error: Data path does not exist: {data_path}")
            print("  Remove --skip-prepare flag to prepare the competition first")
            sys.exit(1)
    else:
        data_path = prepare_competition(competition_id)
        if data_path is None:
            print("\n✗ Failed to prepare competition. Exiting.")
            sys.exit(1)

    # Step 2: Run multi-agent system
    submission_path = run_multi_agent_system(
        competition_id=competition_id,
        data_path=data_path,
        time_budget=args.time_budget
    )

    if submission_path is None:
        print("\n✗ Failed to run multi-agent system. Exiting.")
        sys.exit(1)

    # Step 3: Grade submission (unless skipped)
    if args.skip_grading:
        print("\nSkipping grading step...")
        print(f"Submission saved at: {submission_path}")
    else:
        grade_submission(competition_id, submission_path)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nSubmission file: {submission_path}")
    print(f"Logs directory: /tmp/{competition_id}_logs")
    print()


if __name__ == "__main__":
    main()
