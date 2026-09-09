# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation module for Polaris-Bench model predictions."""

import argparse
import collections
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from evaluation.data_loader import PolarisDataLoader


# Task categories mapping (53 tasks across 5 cognitive categories)
TASK_CATEGORIES = {
    "Algorithmic Logic & Simulation": [
        "sudoku", "four_color", "n_queens", "random_walk", "collision_detection", 
        "bouncing_point"
    ],
    "Combinatorics & Probability": [
        "path_counting", "lattice_paths", "area_counting", "edge_counting", "uncut_cells", 
        "maximum_collection", "longest_path", "largest_number_path", "curve_length"
    ],
    "Navigation & Routing": [
        "maze", "shortest_path", "bounded_path_finding", "wrapping_path_finding", 
        "bounded_diagonal_paths", "wrapping_diagonal_paths", "bounded_knight_paths", 
        "knight_paths", "checkpoint_paths", "monotonic_path", "rule_based_navigation", 
        "absolute_navigation", "egocentric_navigation", "wrapping_navigation"
    ],
    "Spatial Transformation & Geometry": [
        "grid_rotation", "pivot_rotation", "mirror_reflection", "grid_folding", 
        "rotation_center", "rotation_matching", "area_balancing", "minimum_flips", 
        "wall_follower", "letter_collection", "turn_counting", "pipe_lengths", "word_search"
    ],
    "Visual Pattern Matching": [
        "pattern_completion", "pattern_prediction", "layer_completion", "shape_completion", 
        "shape_fitting", "jigsaw_matching", "odd_piece_out", "fragment_matching", 
        "anomaly_detection", "template_matching", "impossible_shape"
    ]
}


def get_task_category(task_name: str) -> str:
    """Returns the cognitive category for a given task name."""
    for category, tasks in TASK_CATEGORIES.items():
        if task_name in tasks:
            return category
    return "Other"


def clean_answer(ans: Any) -> str:
    """Cleans and extracts core answer string from model output or ground truth.
    
    Handles:
    - Markdown styling (e.g. **A**, `A`, *A*)
    - Parentheses or brackets (e.g. (A), [A])
    - Common answer prefixes (e.g. "Answer: A", "The answer is Option A")
    - Trailing periods (e.g. "A.")
    """
    if ans is None:
        return ""
    text = str(ans).strip()
    # Strip markdown quotes and formatting
    text = text.strip("*_`'\" \n\t")

    # If verbose answer like "The correct answer is (A)", extract candidate
    for prefix in ["answer is:", "answer is", "answer:", "choice:", "option:"]:
        if prefix in text.lower():
            text = text.lower().split(prefix)[-1].strip()
            text = text.strip("*_`'\" \n\t")

    # Extract single option letter from formats like "(A)", "[A]", "Option A"
    match = re.search(r"\b(?:option|choice)?\s*[\(\[]?([A-Ha-h])[\)\]]?\.?\b", text)
    if match and len(text) <= 20:
        return match.group(1).upper()

    # Remove enclosing parentheses if present
    if text.startswith("(") and text.endswith(")") and len(text) > 2:
        inner = text[1:-1].strip()
        # Keep coordinate pairs like (7,13)
        if "," not in inner:
            text = inner

    # Remove trailing period
    if text.endswith(".") and not text.replace(".", "", 1).isdigit():
        text = text[:-1].strip()

    return text


def match_answer(prediction: Any, ground_truth: Any) -> bool:
    """Compares prediction and ground truth with robust normalization."""
    if prediction is None or ground_truth is None:
        return False

    pred_str = str(prediction).strip()
    gt_str = str(ground_truth).strip()

    # Direct exact string match
    if pred_str == gt_str:
        return True

    # Normalized match
    clean_pred = clean_answer(pred_str)
    clean_gt = clean_answer(gt_str)

    if clean_pred.upper() == clean_gt.upper():
        return True

    # Numeric equivalence (e.g. "12" vs "12.0")
    try:
        p_num = float(clean_pred)
        g_num = float(clean_gt)
        if abs(p_num - g_num) < 1e-5:
            return True
    except (ValueError, TypeError):
        pass

    # Normalized coordinate tuple matching e.g. "(7,13)" vs "7, 13"
    pred_coords = re.findall(r"-?\d+", clean_pred)
    gt_coords = re.findall(r"-?\d+", clean_gt)
    if pred_coords and gt_coords and pred_coords == gt_coords:
        return True

    return False


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Standard evaluation match function."""
    return match_answer(prediction, ground_truth)


def llm_judge_fallback(prediction: str, ground_truth: str, question: str) -> bool:
    """Optional LLM-as-judge fallback for non-exact semantic matches (stub)."""
    logging.info("LLM Judge fallback stub called.")
    return match_answer(prediction, ground_truth)


class Evaluator:
    """Evaluates model predictions on Polaris-Bench."""

    def __init__(self, ground_truth_data: List[Dict[str, Any]]):
        """Initializes the evaluator with benchmark ground truth records."""
        self.ground_truth: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.by_task_index: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)

        for ex in ground_truth_data:
            task = ex.get("task", "")
            index = str(ex.get("index", ""))
            q_type = ex.get("question_type", "cartesian")
            key = (task, index, q_type)
            self.ground_truth[key] = ex
            self.by_task_index[(task, index)][q_type] = ex

    def _normalize_predictions(
        self, 
        predictions: Any, 
        default_question_type: Optional[str] = None
    ) -> Dict[Tuple[str, str, str], Any]:
        """Normalizes various prediction input formats into {(task, index, question_type): pred}."""
        normalized: Dict[Tuple[str, str, str], Any] = {}

        # Format: list of dicts [{"task": ..., "index": ..., "question_type": ..., "prediction": ...}]
        if isinstance(predictions, list):
            for item in predictions:
                task = item.get("task") or item.get("task_id", "")
                idx = str(item.get("index", ""))
                q_type = (
                    item.get("question_type") 
                    or item.get("type") 
                    or item.get("coordinate_system") 
                    or default_question_type
                )
                pred = item.get("prediction") or item.get("answer") or item.get("pred")
                if task and idx:
                    if q_type:
                        normalized[(task, idx, q_type)] = pred
                    elif (task, idx) in self.by_task_index:
                        # Match available types for this task & index
                        for available_type in self.by_task_index[(task, idx)]:
                            normalized[(task, idx, available_type)] = pred

        # Format: dict
        elif isinstance(predictions, dict):
            for k, v in predictions.items():
                # Nested by question_type e.g. {"cartesian": {...}, "polar": {...}}
                if k in ("cartesian", "polar", "hexagonal", "octagonal") and isinstance(v, dict):
                    q_type = k
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, dict):
                            # {"polar": {"sudoku": {"example_001": "D"}}}
                            task = sub_k
                            for idx, ans in sub_v.items():
                                normalized[(task, str(idx), q_type)] = ans
                        elif "::" in str(sub_k):
                            # {"polar": {"sudoku::example_001": "D"}}
                            parts = str(sub_k).split("::", 1)
                            normalized[(parts[0], str(parts[1]), q_type)] = sub_v
                    continue

                # Nested by task e.g. {"sudoku": {"example_001": "D"}} or {"sudoku": {"polar": {"example_001": "D"}}}
                if isinstance(v, dict):
                    task = k
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, dict):
                            # Could be {"sudoku": {"polar": {"example_001": "D"}}}
                            # or {"sudoku": {"example_001": {"polar": "D", "cartesian": "D"}}}
                            if sub_k in ("cartesian", "polar", "hexagonal", "octagonal"):
                                q_type = sub_k
                                for idx, ans in sub_v.items():
                                    normalized[(task, str(idx), q_type)] = ans
                            else:
                                idx = str(sub_k)
                                for q_type, ans in sub_v.items():
                                    normalized[(task, idx, q_type)] = ans
                        else:
                            # {"sudoku": {"example_001": "D"}}
                            idx = str(sub_k)
                            if default_question_type:
                                normalized[(task, idx, default_question_type)] = sub_v
                            elif (task, idx) in self.by_task_index:
                                for available_type in self.by_task_index[(task, idx)]:
                                    normalized[(task, idx, available_type)] = sub_v
                    continue

                # 3-part key: "sudoku::example_001::polar"
                if isinstance(k, str) and k.count("::") == 2:
                    task, idx, q_type = k.split("::", 2)
                    normalized[(task, str(idx), q_type)] = v
                # 2-part key: "sudoku::example_001"
                elif isinstance(k, str) and "::" in k:
                    task, idx = k.split("::", 1)
                    if default_question_type:
                        normalized[(task, str(idx), default_question_type)] = v
                    elif (task, idx) in self.by_task_index:
                        for available_type in self.by_task_index[(task, idx)]:
                            normalized[(task, str(idx), available_type)] = v
                # Tuple key
                elif isinstance(k, tuple):
                    if len(k) == 3:
                        normalized[(k[0], str(k[1]), k[2])] = v
                    elif len(k) == 2:
                        task, idx = k[0], str(k[1])
                        if default_question_type:
                            normalized[(task, idx, default_question_type)] = v
                        elif (task, idx) in self.by_task_index:
                            for available_type in self.by_task_index[(task, idx)]:
                                normalized[(task, idx, available_type)] = v

        return normalized

    def evaluate(
        self, 
        predictions: Any, 
        question_type: Optional[str] = None,
        use_llm_judge: bool = False
    ) -> Dict[str, Any]:
        """Evaluates predictions and computes benchmark metrics.
        
        Args:
            predictions: Model predictions in flat, nested, or list format.
            question_type: Optional filter to evaluate only a specific coordinate system.
            use_llm_judge: Whether to call LLM judge fallback for non-exact matches.
            
        Returns:
            Structured evaluation report with overall, per-type, per-category,
            and Cartesian-to-Polar accuracy drop metrics.
        """
        flat_preds = self._normalize_predictions(predictions, default_question_type=question_type)

        metrics = {
            "total_evaluated": 0,
            "total_correct": 0,
            "type_total": collections.defaultdict(int),
            "type_correct": collections.defaultdict(int),
            "cat_type_total": collections.defaultdict(lambda: collections.defaultdict(int)),
            "cat_type_correct": collections.defaultdict(lambda: collections.defaultdict(int)),
            "task_type_total": collections.defaultdict(lambda: collections.defaultdict(int)),
            "task_type_correct": collections.defaultdict(lambda: collections.defaultdict(int)),
        }

        for (task, idx, q_type), pred_ans in flat_preds.items():
            if question_type and q_type != question_type:
                continue

            gt_example = self.ground_truth.get((task, idx, q_type))
            if not gt_example:
                continue

            gt_ans = gt_example.get("answer")
            category = get_task_category(task)

            is_correct = match_answer(pred_ans, gt_ans)
            if not is_correct and use_llm_judge:
                question_text = gt_example.get("question", "")
                is_correct = llm_judge_fallback(pred_ans, str(gt_ans), question_text)

            metrics["total_evaluated"] += 1
            metrics["type_total"][q_type] += 1
            metrics["cat_type_total"][category][q_type] += 1
            metrics["task_type_total"][task][q_type] += 1

            if is_correct:
                metrics["total_correct"] += 1
                metrics["type_correct"][q_type] += 1
                metrics["cat_type_correct"][category][q_type] += 1
                metrics["task_type_correct"][task][q_type] += 1

        total = metrics["total_evaluated"]
        correct = metrics["total_correct"]

        # Question type accuracies
        type_acc = {}
        for qt in ["cartesian", "polar", "hexagonal", "octagonal"]:
            t_tot = metrics["type_total"][qt]
            t_cor = metrics["type_correct"][qt]
            if t_tot > 0:
                type_acc[qt] = {
                    "total": t_tot,
                    "correct": t_cor,
                    "accuracy": round(t_cor / t_tot, 4),
                    "accuracy_percent": round((t_cor / t_tot) * 100, 2)
                }

        cart_acc = type_acc.get("cartesian", {}).get("accuracy_percent")
        polar_acc = type_acc.get("polar", {}).get("accuracy_percent")
        drop = round(cart_acc - polar_acc, 2) if (cart_acc is not None and polar_acc is not None) else None

        # Category accuracies
        category_report = {}
        for category in TASK_CATEGORIES:
            cat_types = metrics["cat_type_total"][category]
            cat_tot = sum(cat_types.values())
            cat_cor = sum(metrics["cat_type_correct"][category].values())

            c_cart_tot = cat_types.get("cartesian", 0)
            c_cart_cor = metrics["cat_type_correct"][category].get("cartesian", 0)
            c_cart_acc = round((c_cart_cor / c_cart_tot) * 100, 2) if c_cart_tot > 0 else 0.0

            c_pol_tot = cat_types.get("polar", 0)
            c_pol_cor = metrics["cat_type_correct"][category].get("polar", 0)
            c_pol_acc = round((c_pol_cor / c_pol_tot) * 100, 2) if c_pol_tot > 0 else 0.0

            cat_drop = round(c_cart_acc - c_pol_acc, 2) if (c_cart_tot > 0 and c_pol_tot > 0) else 0.0

            category_report[category] = {
                "total": cat_tot,
                "overall_accuracy": round((cat_cor / cat_tot) * 100, 2) if cat_tot > 0 else 0.0,
                "cartesian_accuracy": c_cart_acc,
                "polar_accuracy": c_pol_acc,
                "cartesian_to_polar_drop": cat_drop
            }

        # Task accuracies
        task_report = {}
        for task in sorted(metrics["task_type_total"].keys()):
            t_types = metrics["task_type_total"][task]
            t_tot = sum(t_types.values())
            t_cor = sum(metrics["task_type_correct"][task].values())

            t_cart_tot = t_types.get("cartesian", 0)
            t_cart_cor = metrics["task_type_correct"][task].get("cartesian", 0)
            t_cart_acc = round((t_cart_cor / t_cart_tot) * 100, 2) if t_cart_tot > 0 else None

            t_pol_tot = t_types.get("polar", 0)
            t_pol_cor = metrics["task_type_correct"][task].get("polar", 0)
            t_pol_acc = round((t_pol_cor / t_pol_tot) * 100, 2) if t_pol_tot > 0 else None

            t_drop = round(t_cart_acc - t_pol_acc, 2) if (t_cart_acc is not None and t_pol_acc is not None) else None

            task_report[task] = {
                "total": t_tot,
                "cartesian_accuracy": t_cart_acc,
                "polar_accuracy": t_pol_acc,
                "cartesian_to_polar_drop": t_drop
            }

        report = {
            "summary": {
                "total_evaluated": total,
                "overall_accuracy_percent": round((correct / total) * 100, 2) if total > 0 else 0.0,
                "cartesian_accuracy_percent": cart_acc,
                "polar_accuracy_percent": polar_acc,
                "cartesian_to_polar_drop": drop
            },
            "by_question_type": type_acc,
            "by_category": category_report,
            "by_task": task_report
        }
        return report


def format_ascii_table(report: Dict[str, Any]) -> str:
    """Formats the evaluation summary into a clean ASCII table."""
    summary = report.get("summary", {})
    by_cat = report.get("by_category", {})

    lines = []
    lines.append("=" * 82)
    lines.append("                         POLARIS-BENCH EVALUATION REPORT")
    lines.append("=" * 82)
    
    total = summary.get("total_evaluated", 0)
    overall = summary.get("overall_accuracy_percent", 0.0)
    cart = summary.get("cartesian_accuracy_percent", "N/A")
    pol = summary.get("polar_accuracy_percent", "N/A")
    drop = summary.get("cartesian_to_polar_drop", "N/A")

    lines.append(f"Total Evaluated: {total} | Overall Accuracy: {overall}%")
    lines.append(f"Cartesian Acc:   {cart}% | Polar Acc: {pol}% | Drop (Cartesian - Polar): {drop} pt")
    lines.append("-" * 82)
    lines.append(f"{'Category':<36} | {'Cartesian (%)':<13} | {'Polar (%)':<10} | {'Drop (pt)':<10}")
    lines.append("-" * 82)

    for cat_name, cat_data in by_cat.items():
        c_acc = f"{cat_data.get('cartesian_accuracy', 0.0):.1f}"
        p_acc = f"{cat_data.get('polar_accuracy', 0.0):.1f}"
        d_val = f"{cat_data.get('cartesian_to_polar_drop', 0.0):.1f}"
        lines.append(f"{cat_name:<36} | {c_acc:<13} | {p_acc:<10} | {d_val:<10}")

    lines.append("=" * 82)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Polaris-Bench")
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to JSON predictions file."
    )
    parser.add_argument(
        "--ground-truth", type=str, required=False, default=None,
        help="Path to Polaris-Bench JSON file. If omitted, loads from HF or sample_tasks."
    )
    parser.add_argument(
        "--question-type", type=str, required=False, default=None,
        choices=["cartesian", "polar", "hexagonal", "octagonal"],
        help="Optional filter to evaluate a specific coordinate system."
    )
    parser.add_argument(
        "--output-report", type=str, default="evaluation_report.json",
        help="Path to save evaluation output JSON (default: evaluation_report.json)."
    )
    parser.add_argument(
        "--use-llm-judge", action="store_true",
        help="Enable LLM judge fallback for exact-match mismatches (stub)."
    )
    
    args = parser.parse_args()
    
    print(f"Loading ground truth data...")
    data_loader = PolarisDataLoader(args.ground_truth)
    print(f"Loaded {len(data_loader.data)} examples.")
    
    print(f"Loading predictions from {args.predictions}...")
    with open(args.predictions, "r", encoding="utf-8") as f:
        preds = json.load(f)
        
    evaluator = Evaluator(data_loader.data)
    report = evaluator.evaluate(
        preds, 
        question_type=args.question_type, 
        use_llm_judge=args.use_llm_judge
    )
    
    print("\n" + format_ascii_table(report))
    
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"Detailed JSON report saved to: {args.output_report}\n")


if __name__ == "__main__":
    main()

