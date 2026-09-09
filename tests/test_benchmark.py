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

"""Unit tests for Polaris-Bench data loader and evaluation module."""

import unittest
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluation.data_loader import PolarisDataLoader
from evaluation.evaluate import (
    Evaluator,
    clean_answer,
    match_answer,
    get_task_category,
    format_ascii_table,
    TASK_CATEGORIES
)


class TestPolarisBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = os.path.join(ROOT_DIR, "examples/sample_tasks/sample_tasks.json")
        cls.loader = PolarisDataLoader(cls.sample_path)

    def test_data_loader_loading(self):
        self.assertEqual(len(self.loader), 40)
        # Check that both Cartesian and Polar representations are present
        cart_samples = self.loader.get_dataset(question_type="cartesian")
        polar_samples = self.loader.get_dataset(question_type="polar")
        self.assertEqual(len(cart_samples), 20)
        self.assertEqual(len(polar_samples), 20)

    def test_paired_dataset(self):
        pairs = self.loader.get_paired_dataset()
        self.assertEqual(len(pairs), 20)
        for cart, pol in pairs:
            self.assertEqual(cart["task"], pol["task"])
            self.assertEqual(cart["index"], pol["index"])
            self.assertEqual(cart["question_type"], "cartesian")
            self.assertEqual(pol["question_type"], "polar")

    def test_clean_answer(self):
        self.assertEqual(clean_answer("(A)"), "A")
        self.assertEqual(clean_answer("[B]"), "B")
        self.assertEqual(clean_answer("Option C"), "C")
        self.assertEqual(clean_answer("Answer: D"), "D")
        self.assertEqual(clean_answer("**E**"), "E")
        self.assertEqual(clean_answer("A."), "A")
        self.assertEqual(clean_answer("(7, 13)"), "(7, 13)")

    def test_match_answer(self):
        self.assertTrue(match_answer("A", "A"))
        self.assertTrue(match_answer("(A)", "A"))
        self.assertTrue(match_answer("The answer is: A", "A"))
        self.assertTrue(match_answer("12.0", "12"))
        self.assertTrue(match_answer("(7, 13)", "(7,13)"))
        self.assertFalse(match_answer("A", "B"))

    def test_evaluator_3part_keys(self):
        evaluator = Evaluator(self.loader.data)
        # All correct for Cartesian, all wrong for Polar
        preds = {}
        for ex in self.loader.data:
            key = f"{ex["task"]}::{ex["index"]}::{ex["question_type"]}"
            if ex["question_type"] == "cartesian":
                preds[key] = ex["answer"]
            else:
                preds[key] = "WRONG_ANSWER"

        report = evaluator.evaluate(preds)
        summary = report["summary"]
        self.assertEqual(summary["total_evaluated"], 40)
        self.assertEqual(summary["cartesian_accuracy_percent"], 100.0)
        self.assertEqual(summary["polar_accuracy_percent"], 0.0)
        self.assertEqual(summary["cartesian_to_polar_drop"], 100.0)

    def test_evaluator_nested_dict(self):
        evaluator = Evaluator(self.loader.data)
        preds = {
            "cartesian": {},
            "polar": {}
        }
        for ex in self.loader.data:
            t = ex["task"]
            idx = ex["index"]
            qt = ex["question_type"]
            preds[qt][f"{t}::{idx}"] = ex["answer"]

        report = evaluator.evaluate(preds)
        summary = report["summary"]
        self.assertEqual(summary["cartesian_accuracy_percent"], 100.0)
        self.assertEqual(summary["polar_accuracy_percent"], 100.0)
        self.assertEqual(summary["cartesian_to_polar_drop"], 0.0)

    def test_evaluator_list_format(self):
        evaluator = Evaluator(self.loader.data)
        pred_list = []
        for ex in self.loader.data:
            pred_list.append({
                "task": ex["task"],
                "index": ex["index"],
                "question_type": ex["question_type"],
                "prediction": ex["answer"]
            })
        report = evaluator.evaluate(pred_list)
        self.assertEqual(report["summary"]["overall_accuracy_percent"], 100.0)

    def test_categories_coverage(self):
        self.assertEqual(len(TASK_CATEGORIES), 5)
        for cat, tasks in TASK_CATEGORIES.items():
            self.assertGreater(len(tasks), 0)
        cat = get_task_category("sudoku")
        self.assertEqual(cat, "Algorithmic Logic & Simulation")

    def test_format_ascii_table(self):
        evaluator = Evaluator(self.loader.data)
        preds = {f"{ex["task"]}::{ex["index"]}::{ex["question_type"]}": ex["answer"] for ex in self.loader.data}
        report = evaluator.evaluate(preds)
        table = format_ascii_table(report)
        self.assertIn("POLARIS-BENCH EVALUATION REPORT", table)
        self.assertIn("Algorithmic Logic & Simulation", table)


if __name__ == "__main__":
    unittest.main()
