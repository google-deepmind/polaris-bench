"""Quick start example for Polaris-Bench dataset.

Demonstrates loading the dataset (from local sample tasks or HuggingFace),
paired Cartesian vs. Polar inspection, and evaluating predictions.
"""

import argparse
import os
import sys

# Ensure repository root is in sys.path when running as a standalone script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluation.data_loader import PolarisDataLoader
from evaluation.evaluate import Evaluator, format_ascii_table


def main():
    parser = argparse.ArgumentParser(description="Polaris-Bench Quick Start")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to local dataset JSON (e.g., polaris_bench.json or sample_tasks.json)."
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Force loading from HuggingFace Hub ('google/polaris-bench')."
    )
    args = parser.parse_args()

    # Determine default data source
    data_path = args.data_path
    if not args.hf and data_path is None:
        # Check standard local paths
        candidates = [
            "examples/sample_tasks/sample_tasks.json",
            "polaris_bench.json",
            "../examples/sample_tasks/sample_tasks.json"
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    if data_path:
        print(f"Loading Polaris-Bench data from local file: {data_path} ...")
        loader = PolarisDataLoader(data_path)
    else:
        print("Loading Polaris-Bench dataset from HuggingFace Hub ('google/polaris-bench')...")
        loader = PolarisDataLoader()

    print(f"Successfully loaded {len(loader)} total evaluation instances.\n")

    # 1. Filter by task and coordinate system
    target_task = "sudoku"
    target_type = "cartesian"
    filtered = loader.get_dataset(task=target_task, question_type=target_type)
    print(f"Filter query: task='{target_task}', question_type='{target_type}'")
    print(f"Found {len(filtered)} matching instance(s).")

    if filtered:
        sample = filtered[0]
        print("\n--- Example Instance Details ---")
        print(f"Task:          {sample.get('task')}")
        print(f"Index:         {sample.get('index')}")
        print(f"Coordinate:    {sample.get('question_type')}")
        print(f"Image Path:    {sample.get('image_path')}")
        print(f"Answer:        {sample.get('answer')}")
        print(f"Question Preview:\n{sample.get('question', '')[:140]}...\n")

    # 2. Inspect Paired Tasks (Cartesian vs Polar)
    print("--- Paired Cartesian vs. Polar Evaluation ---")
    pairs = loader.get_paired_dataset(task=target_task)
    if pairs:
        cart_ex, pol_ex = pairs[0]
        print(f"Task: {cart_ex.get('task')} | Index: {cart_ex.get('index')}")
        print(f"  [Cartesian] Image: {cart_ex.get('image_path')}")
        print(f"  [Polar]     Image: {pol_ex.get('image_path')}")
        print(f"  Ground-Truth Answer: {cart_ex.get('answer')}\n")

    # 3. Quick Evaluation Demo
    print("--- Running Demo Evaluation ---")
    evaluator = Evaluator(loader.data)
    # Mock dummy predictions
    mock_predictions = {}
    for ex in loader.data:
        key = f"{ex.get('task')}::{ex.get('index')}::{ex.get('question_type')}"
        # Cartesian gets correct answer, Polar gets alternative choice
        if ex.get("question_type") == "cartesian":
            mock_predictions[key] = ex.get("answer")
        else:
            mock_predictions[key] = "X"

    report = evaluator.evaluate(mock_predictions)
    print(format_ascii_table(report))
    print("Quick start completed successfully!")


if __name__ == "__main__":
    main()

