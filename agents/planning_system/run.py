#!/usr/bin/env python
"""
Standalone runner for the agentic planning system

This script can be run directly without module path issues.
"""

import sys
from pathlib import Path

# Add mle-bench to Python path
mle_bench_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(mle_bench_dir))

# Now import and run the CLI
from agents.planning_system.cli import main

if __name__ == "__main__":
    main()
