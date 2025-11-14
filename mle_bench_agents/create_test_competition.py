"""
Create a simple test competition with synthetic data.

Generates train.csv, test.csv, and sample_submission.csv for testing the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification


def create_test_competition(output_dir: Path = Path("/home/data"), n_samples: int = 1000):
    """
    Create synthetic competition data.

    Args:
        output_dir: Directory to save files
        n_samples: Number of samples to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating test competition data in {output_dir}")
    print(f"Generating {n_samples} samples...")

    # Generate classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.1  # Add some noise
    )

    # Split into train and test
    train_size = int(0.7 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    # Create train.csv
    train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X.shape[1])])
    train_df["target"] = y_train
    train_df["id"] = range(len(train_df))

    # Reorder columns (id first, target last)
    cols = ["id"] + [f"feature_{i}" for i in range(X.shape[1])] + ["target"]
    train_df = train_df[cols]

    train_path = output_dir / "train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"✓ Created train.csv: {train_df.shape}")

    # Create test.csv (without target)
    test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X.shape[1])])
    test_df["id"] = range(len(test_df))

    # Reorder columns (id first)
    cols = ["id"] + [f"feature_{i}" for i in range(X.shape[1])]
    test_df = test_df[cols]

    test_path = output_dir / "test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"✓ Created test.csv: {test_df.shape}")

    # Create sample_submission.csv
    sample_sub = pd.DataFrame({
        "id": test_df["id"].values,
        "target": np.zeros(len(test_df))  # Placeholder predictions
    })

    sample_sub_path = output_dir / "sample_submission.csv"
    sample_sub.to_csv(sample_sub_path, index=False)
    print(f"✓ Created sample_submission.csv: {sample_sub.shape}")

    # Save ground truth for validation (not accessible to agents)
    ground_truth_df = pd.DataFrame({
        "id": test_df["id"].values,
        "target": y_test
    })

    ground_truth_path = output_dir / "ground_truth.csv"
    ground_truth_df.to_csv(ground_truth_path, index=False)
    print(f"✓ Created ground_truth.csv (for evaluation): {ground_truth_df.shape}")

    # Create description file
    description = """
# Test Competition - Binary Classification

## Task
Predict the binary target variable (0 or 1) for test samples.

## Data
- **train.csv**: Training data with features and target
- **test.csv**: Test data (predict target for these samples)
- **sample_submission.csv**: Example submission format

## Features
- 10 numerical features (feature_0 to feature_9)
- Binary target (0 or 1)

## Evaluation
- Metric: Accuracy
- Direction: Maximize

## Submission Format
CSV with columns: id, target
"""

    desc_path = output_dir / "description.txt"
    with open(desc_path, 'w') as f:
        f.write(description)
    print(f"✓ Created description.txt")

    print("\nTest competition created successfully!")
    print(f"\nFiles created:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {sample_sub_path}")
    print(f"  - {ground_truth_path}")
    print(f"  - {desc_path}")

    return output_dir


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Create test competition data")
    parser.add_argument("--output-dir", type=str, default="/home/data", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")

    args = parser.parse_args()

    create_test_competition(
        output_dir=Path(args.output_dir),
        n_samples=args.n_samples
    )


if __name__ == "__main__":
    main()
