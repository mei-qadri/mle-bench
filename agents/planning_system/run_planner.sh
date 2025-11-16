#!/bin/bash
#
# Helper script to run the agentic planning system CLI
#
# Usage:
#   ./run_planner.sh plan <competition_id>
#   ./run_planner.sh run <competition_id>
#   ./run_planner.sh execute <plan_file>

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MLE_BENCH_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Set PYTHONPATH to include mle-bench directory
export PYTHONPATH="${MLE_BENCH_DIR}:${PYTHONPATH}"

# Run the CLI module
cd "${MLE_BENCH_DIR}"
python -m agents.planning_system.cli "$@"
