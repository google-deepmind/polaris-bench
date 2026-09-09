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

"""Data loader for Polaris-Bench evaluation dataset."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class PolarisDataLoader:
    """Loader for the Polaris-Bench dataset.

    Supports loading from:
    1. HuggingFace Hub ("google/polaris-bench")
    2. Local flat benchmark JSON (e.g., full 10,800 problems polaris_bench.json)
    3. Local paired benchmark JSON (e.g., examples/sample_tasks/sample_tasks.json)
    """

    def __init__(self, data_path: Optional[str] = None):
        """Initializes the data loader.

        Args:
            data_path: Optional path to a local JSON dataset file. If not
              provided, the dataset will be loaded from the HuggingFace Hub.
        """
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Loads data from the specified local file or from HuggingFace Hub."""
        if self.data_path:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(
                    f"Dataset file not found at: {self.data_path}"
                )

            with open(self.data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if isinstance(raw_data, list):
                # Detect if the file is in paired format (like sample_tasks.json)
                if (
                    raw_data
                    and "cartesian" in raw_data[0]
                    and "polar" in raw_data[0]
                ):
                    flattened = []
                    for item in raw_data:
                        task_id = item.get("id") or item.get("task", "")
                        index = str(item.get("index", ""))
                        answer = str(item.get("answer", ""))
                        category = item.get("category_name") or item.get(
                            "category", ""
                        )

                        # Add Cartesian instance
                        cart = item["cartesian"]
                        flattened.append({
                            "task": task_id,
                            "index": index,
                            "question_type": "cartesian",
                            "question": cart.get("question", ""),
                            "answer": answer,
                            "image_path": cart.get("gh_image")
                            or cart.get("image", ""),
                            "category": category,
                        })

                        # Add Polar instance
                        polar = item["polar"]
                        flattened.append({
                            "task": task_id,
                            "index": index,
                            "question_type": "polar",
                            "question": polar.get("question", ""),
                            "answer": answer,
                            "image_path": polar.get("gh_image")
                            or polar.get("image", ""),
                            "category": category,
                        })
                    return flattened

                return raw_data
            else:
                raise ValueError(
                    f"Expected list of records in {self.data_path}, got {type(raw_data)}"
                )
        else:
            # Lazy import to avoid hard dependency on 'datasets' for local JSON users
            try:
                import datasets
            except ImportError:
                raise ImportError(
                    "Loading from HuggingFace requires 'datasets'. "
                    "Please install with: pip install datasets"
                )

            hf_dataset = datasets.load_dataset(
                "google/polaris-bench", split="test"
            )
            return list(hf_dataset)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def get_dataset(
        self,
        task: Optional[str] = None,
        question_type: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves dataset optionally filtered by task, question type, or category.

        Args:
            task: Task name to filter by (e.g., 'sudoku', 'maze').
            question_type: Coordinate system type to filter by ('cartesian',
              'polar', 'hexagonal', 'octagonal').
            category: Cognitive category name to filter by.

        Returns:
            A list of examples matching the filters.
        """
        filtered_data = self.data
        if task:
            filtered_data = [
                ex for ex in filtered_data if ex.get("task") == task
            ]
        if question_type:
            filtered_data = [
                ex
                for ex in filtered_data
                if ex.get("question_type") == question_type
            ]
        if category:
            filtered_data = [
                ex for ex in filtered_data if ex.get("category") == category
            ]
        return filtered_data

    def get_paired_dataset(
        self, task: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Retrieves paired (Cartesian, Polar) instances sharing the same task and index.

        Args:
            task: Optional task name to filter by.

        Returns:
            A list of tuples: (cartesian_example, polar_example).
        """
        cartesian_map = {}
        polar_map = {}

        for ex in self.data:
            t = ex.get("task")
            if task and t != task:
                continue
            idx = str(ex.get("index"))
            q_type = ex.get("question_type")

            if q_type == "cartesian":
                cartesian_map[(t, idx)] = ex
            elif q_type == "polar":
                polar_map[(t, idx)] = ex

        pairs = []
        for key, cart_ex in cartesian_map.items():
            if key in polar_map:
                pairs.append((cart_ex, polar_map[key]))

        return pairs

    def load_image(
        self, example: Dict[str, Any], base_dir: Optional[str] = None
    ):
        """Loads the image associated with an example.

        Returns a PIL Image object.
        """
        if "image" in example and example["image"] is not None:
            return example["image"]

        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Image loading requires Pillow. Please install with: pip"
                " install Pillow"
            )

        img_path = example.get("image_path", "")
        if base_dir:
            full_path = os.path.join(base_dir, img_path)
        else:
            full_path = img_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at: {full_path}")

        return Image.open(full_path)

