"""
End-to-end test for MLE-bench Multi-Agent System.

Tests the complete pipeline on synthetic data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mle_bench_agents.create_test_competition import create_test_competition
from mle_bench_agents.runner import CompetitionRunner


def test_end_to_end():
    """Test complete pipeline."""
    print("="*80)
    print("MLE-BENCH MULTI-AGENT SYSTEM - END-TO-END TEST")
    print("="*80)

    # Setup paths
    data_dir = Path("/tmp/test_competition_data")
    submission_dir = Path("/tmp/test_competition_submission")
    log_dir = Path("/tmp/test_competition_logs")

    # Clean up previous runs
    import shutil
    for dir_path in [data_dir, submission_dir, log_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)

    # Step 1: Create test competition
    print("\n" + "-"*80)
    print("STEP 1: Creating Test Competition Data")
    print("-"*80)

    create_test_competition(output_dir=data_dir, n_samples=1000)

    # Verify files exist
    required_files = ["train.csv", "test.csv", "sample_submission.csv"]
    for filename in required_files:
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"❌ ERROR: Required file {filename} not found!")
            return False

    print("\n✓ All required files created")

    # Step 2: Run competition
    print("\n" + "-"*80)
    print("STEP 2: Running Multi-Agent System")
    print("-"*80)

    runner = CompetitionRunner(
        log_dir=log_dir,
        log_level="INFO"
    )

    result = runner.run_competition(
        competition_id="test-competition",
        data_path=data_dir,
        submission_path=submission_dir,
        time_budget=300  # 5 minutes for testing
    )

    # Step 3: Validate results
    print("\n" + "-"*80)
    print("STEP 3: Validating Results")
    print("-"*80)

    if result.get("status") != "success":
        print(f"❌ TEST FAILED: {result.get('error', 'Unknown error')}")
        return False

    # Check submission file exists
    submission_path = Path(result.get("submission_path"))
    if not submission_path.exists():
        print(f"❌ TEST FAILED: Submission file not created")
        return False

    print(f"✓ Submission file created: {submission_path}")

    # Load and validate submission
    import pandas as pd
    submission_df = pd.read_csv(submission_path)
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")

    if submission_df.shape != sample_sub.shape:
        print(f"❌ TEST FAILED: Submission shape mismatch")
        print(f"   Expected: {sample_sub.shape}, Got: {submission_df.shape}")
        return False

    print(f"✓ Submission format correct: {submission_df.shape}")

    # Check for NaN/inf
    if submission_df.isnull().any().any():
        print(f"❌ TEST FAILED: Submission contains NaN values")
        return False

    print(f"✓ No NaN values in submission")

    # Calculate accuracy on test set
    ground_truth = pd.read_csv(data_dir / "ground_truth.csv")
    predictions = submission_df["target"].values
    true_labels = ground_truth["target"].values

    # Convert predictions to binary if needed
    if predictions.dtype == float:
        predictions = (predictions > 0.5).astype(int)

    accuracy = (predictions == true_labels).mean()
    print(f"\n✓ Test Accuracy: {accuracy:.4f}")

    # Check if accuracy is reasonable (better than random)
    if accuracy < 0.6:
        print(f"⚠ WARNING: Accuracy seems low ({accuracy:.4f}), but pipeline completed")

    # Step 4: Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✓ Status: {result['status']}")
    print(f"✓ Submission Path: {result['submission_path']}")
    print(f"✓ CV Score: {result.get('best_cv_score', 'N/A')}")
    print(f"✓ Models Trained: {result.get('n_models', 0)}")
    print(f"✓ Ensemble Used: {result.get('ensemble_used', False)}")
    print(f"✓ Test Accuracy: {accuracy:.4f}")
    print()
    print("="*80)
    print("✅ END-TO-END TEST PASSED!")
    print("="*80)

    return True


def main():
    """Main entry point."""
    try:
        success = test_end_to_end()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
