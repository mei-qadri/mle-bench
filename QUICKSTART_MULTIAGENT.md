# Quick Start Guide: Testing the Multi-Agent System

This guide will help you quickly test the multi-agent system on MLE-bench competitions.

## Prerequisites

1. **Install MLE-bench** (if not already done):
   ```bash
   pip install -e .
   ```

2. **Install multi-agent dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm pyyaml psutil
   ```

3. **Kaggle API credentials** (optional, only needed if downloading raw data):
   - Download your `kaggle.json` from Kaggle
   - Place it in `~/.kaggle/kaggle.json`

## Quick Test (Recommended)

The fastest way to test is with a **low complexity tabular competition**:

```bash
# 1. List available low complexity competitions
python run_competition.py --list --lite

# 2. Run on a small tabular competition (takes ~5 minutes)
python run_competition.py --competition nomad2018-predict-transparent-conductors
```

This will automatically:
- ✓ Prepare the competition data
- ✓ Run all 5 phases of the multi-agent system
- ✓ Generate a submission file
- ✓ Grade the submission and show your score

## Step-by-Step Instructions

### Step 1: List Available Competitions

```bash
# See all 75 competitions
python run_competition.py --list

# See only 22 low complexity competitions (recommended for testing)
python run_competition.py --list --lite
```

### Step 2: Choose a Competition

**Recommended competitions for first test:**

| Competition ID | Type | Size | Description |
|----------------|------|------|-------------|
| `nomad2018-predict-transparent-conductors` | Tabular | 6 MB | Predict material properties |
| `tabular-playground-series-dec-2021` | Tabular | 700 MB | Multi-class classification |
| `detecting-insults-in-social-commentary` | Text | 2 MB | Binary text classification |
| `random-acts-of-pizza` | Text | 3 MB | Text classification |

### Step 3: Run the Multi-Agent System

```bash
# Basic usage (5 minute time budget)
python run_competition.py --competition nomad2018-predict-transparent-conductors

# With custom time budget (10 minutes)
python run_competition.py --competition tabular-playground-series-dec-2021 --time-budget 600

# If competition is already prepared, skip preparation
python run_competition.py --competition nomad2018-predict-transparent-conductors --skip-prepare
```

### Step 4: Review Results

The script will output:
- **Status**: success/failure
- **CV Score**: Cross-validation score from training
- **Models Trained**: Number of models trained
- **Grading Score**: Official competition metric score

Files created:
- **Submission**: `/tmp/<competition-id>_submission/submission.csv`
- **Logs**: `/tmp/<competition-id>_logs/`

## Understanding the Multi-Agent System

The system runs through **5 phases**:

1. **Understanding (10%)** - Agent A2 analyzes the competition data
   - Identifies task type (classification/regression)
   - Assesses complexity (low/medium/high)
   - Generates recommendations

2. **Preparation (15%)** - Agent A4 engineers features
   - Handles missing values
   - Encodes categorical features
   - Creates cross-validation folds

3. **Modeling (60%)** - Agent A3c trains models
   - Trains multiple models (XGBoost, LightGBM, RandomForest)
   - Evaluates with cross-validation
   - Selects best models

4. **Ensembling (10%)** - Agent A5 combines predictions
   - Weighted averaging based on CV scores
   - Optimizes ensemble performance

5. **Submission (5%)** - Agent A6 validates output
   - Validates submission format
   - Auto-fixes common issues (NaN, inf values)
   - Saves final submission

## Interpreting Results

### Cross-Validation (CV) Score
- Shows how well models performed on training data
- Lower is better for regression (RMSE/MAE)
- Higher is better for classification (Accuracy/AUC)

### Grading Score
- Official competition metric on held-out test set
- Compare to leaderboard to see how you rank
- Good scores indicate the agents are working correctly

### What to Expect

**Low Complexity Competitions:**
- Should complete in 5-10 minutes
- CV score should be reasonable (not random)
- Grading score should beat simple baselines

**Medium/High Complexity:**
- May need longer time budgets (30-60 minutes)
- May require specialized agents (CV/NLP)
- Current system optimized for tabular tasks

## Troubleshooting

### Competition preparation fails
```bash
# Make sure you have Kaggle credentials
ls ~/.kaggle/kaggle.json

# Try preparing manually
mlebench prepare -c <competition-id>
```

### Agent fails during execution
```bash
# Check logs for detailed error messages
cat /tmp/<competition-id>_logs/competition_runner.log

# Common issues:
# - Missing dependencies: install xgboost, lightgbm
# - Memory issues: reduce time budget or use smaller competition
# - Data format issues: check train.csv and test.csv exist
```

### Grading fails
```bash
# Check submission file exists and has correct format
head /tmp/<competition-id>_submission/submission.csv

# Compare to sample submission
head ~/.cache/mlebench/<competition-id>/prepared/public/sample_submission.csv

# Grade manually
mlebench grade-sample /tmp/<competition-id>_submission/submission.csv <competition-id>
```

## Advanced Usage

### Running specific phases only

Edit `mle_bench_agents/runner.py` to skip phases you don't want to run.

### Testing on custom data

```python
from mle_bench_agents.runner import CompetitionRunner

runner = CompetitionRunner(log_dir="./logs", log_level="DEBUG")
result = runner.run_competition(
    competition_id="my-competition",
    data_path="/path/to/data",
    submission_path="./submissions",
    time_budget=600
)
```

### Adding new agents

See `docs/multi_agent_system_design.md` for the full architecture.

To add a new specialist agent:
1. Create agent in `mle_bench_agents/agents/specialists/`
2. Register in `mle_bench_agents/runner.py`
3. Add prompts in `docs/prompts/`

## Next Steps

1. **Test on more competitions**: Try different types (text, image, audio)
2. **Improve agents**: Add hyperparameter optimization, better feature engineering
3. **Add specialists**: Create CV/NLP/Audio specialist agents
4. **Tune parameters**: Adjust time budgets, model selection, ensemble strategies

## Getting Help

- **Documentation**: See `docs/` directory for full system design
- **Examples**: See `mle_bench_agents/examples.py` for code samples
- **Tests**: Run `pytest mle_bench_agents/tests/` to verify installation

## Benchmarking Your System

To compare with the MLE-bench leaderboard:

1. Run on the full **low complexity split** (22 competitions)
2. Use standard time budget: **24 hours per competition** (86400s)
3. Run with **3 different random seeds**
4. Report: mean ± standard error

```bash
# Example benchmark run
for comp in $(cat experiments/splits/low.txt); do
  python run_competition.py --competition $comp --time-budget 86400
done
```

Current leaderboard leaders achieve:
- **Low complexity**: 48-68% success rate
- **All competitions**: 16-48% success rate

The multi-agent system is currently optimized for tabular tasks. To compete on the full benchmark, you'll need to implement specialist agents for CV, NLP, and audio tasks as described in the design documentation.
