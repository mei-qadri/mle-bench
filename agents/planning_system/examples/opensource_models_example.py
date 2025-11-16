#!/usr/bin/env python
"""
Open-Source Models Example

Demonstrates how to use the agentic planning system with open-source models
from Hugging Face instead of commercial APIs.

Supported models:
- GPT OSS 20B/120B (reasoning models with mxfp4 quantization)
- Llama 3.1 8B Instruct
- Mistral 7B Instruct
- Qwen 2.5 7B Instruct

Requirements:
    pip install transformers accelerate torch
    pip install triton>=3.4  # For mxfp4 support
"""

import sys
from pathlib import Path

# Add mle-bench to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agents.planning_system.planning.planner import (
    PlanningModule,
    ProblemSpecification,
)
from agents.planning_system.core.llm import (
    get_gpt_oss_20b_config,
    get_llama_3_8b_config,
    get_mistral_7b_config,
    get_qwen_7b_config,
)


def main():
    """Demonstrate using open-source models"""

    print("\n" + "="*60)
    print("Agentic Planning System - Open-Source Models Example")
    print("="*60 + "\n")

    # Step 1: Choose a model
    print("Available open-source models:")
    print("  1. GPT OSS 20B (reasoning model, 16GB RAM with mxfp4)")
    print("  2. Llama 3.1 8B Instruct (general purpose)")
    print("  3. Mistral 7B Instruct (efficient)")
    print("  4. Qwen 2.5 7B Instruct (multilingual)")
    print()

    model_choice = input("Select a model (1-4, or press Enter for default): ").strip()

    # Get model configuration
    if model_choice == "1":
        llm_config = get_gpt_oss_20b_config()
        print("\n✓ Using GPT OSS 20B (requires transformers, triton>=3.4)")
    elif model_choice == "2":
        llm_config = get_llama_3_8b_config(load_in_4bit=True)
        print("\n✓ Using Llama 3.1 8B Instruct (4-bit quantized)")
    elif model_choice == "3":
        llm_config = get_mistral_7b_config(load_in_4bit=True)
        print("\n✓ Using Mistral 7B Instruct (4-bit quantized)")
    elif model_choice == "4":
        llm_config = get_qwen_7b_config(load_in_4bit=True)
        print("\n✓ Using Qwen 2.5 7B Instruct (4-bit quantized)")
    else:
        llm_config = get_llama_3_8b_config(load_in_4bit=True)
        print("\n✓ Using default: Llama 3.1 8B Instruct (4-bit)")

    print(f"   Model: {llm_config.model_name}")
    print(f"   Provider: {llm_config.provider}")
    print()

    # Step 2: Define the problem
    problem_spec = ProblemSpecification(
        task_description="Binary classification: Predict passenger transport to alternate dimension",
        competition_id="spaceship-titanic",
        data_files=["train.csv", "test.csv"],
        evaluation_metric="accuracy",
        domain="tabular",
        difficulty="low",
    )

    # Step 3: Create planning module with custom LLM
    print("Creating planning module with open-source model...")
    planner = PlanningModule(llm_config=llm_config)

    # Step 4: Generate plan (uses open-source model for planning)
    print("\nGenerating workflow plan...")
    print("Note: First run will download the model (this may take time)")
    print()

    # Check if we should actually run this
    run_planning = input("Proceed with model download and planning? (y/n): ").strip().lower()

    if run_planning == "y":
        try:
            plan = planner.plan_workflow(problem_spec)

            print("\n" + "="*60)
            print("Generated Workflow Plan")
            print("="*60 + "\n")
            print(plan.visualize())

            print("\n" + "="*60)
            print("Success!")
            print("="*60 + "\n")

            print("The workflow was planned using an open-source model!")
            print("This demonstrates that you can use the agentic planning system")
            print("without commercial API keys.\n")

        except ImportError as e:
            print(f"\n⚠️  Missing dependencies: {e}")
            print("\nTo use Hugging Face models, install:")
            print("  pip install transformers accelerate torch")
            print("  pip install triton>=3.4  # For GPT OSS models with mxfp4")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nMake sure you have:")
            print("  1. Sufficient GPU/RAM for the model")
            print("  2. All required dependencies installed")
            print("  3. Hugging Face token (if model requires authentication)")
            print("     export HF_TOKEN='your-token-here'")

    else:
        print("\nSkipping model download and planning.")
        print("\nTo try this later, run:")
        print("  python agents/planning_system/examples/opensource_models_example.py")

    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60 + "\n")

    print("Next steps:")
    print("  1. Explore different open-source models")
    print("  2. Compare planning quality vs commercial models")
    print("  3. Use quantization (4-bit/8-bit) to reduce memory")
    print("  4. Run full workflows with open-source models")
    print("\nBenefits of open-source models:")
    print("  ✓ No API costs")
    print("  ✓ Full privacy (runs locally)")
    print("  ✓ No rate limits")
    print("  ✓ Offline operation")
    print("  ✓ Customizable (fine-tuning)")


if __name__ == "__main__":
    main()
