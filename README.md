<p align="center">
  <img src="docs/static/figures/intro.png" width="100%" alt="Polaris-Bench">
</p>

<h1 align="center">Polaris-Bench</h1>

<p align="center">
  <strong>The Cartesian Shortcut: Re-evaluate Vision Reasoning in Polar Coordinate Space</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.09883"><img src="https://img.shields.io/badge/arXiv-2605.09883-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/google/polaris-bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg" alt="HuggingFace"></a>
  <a href="https://google-deepmind.github.io/polaris-bench"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-Apache%202.0-green.svg" alt="License"></a>
  <a href="https://huggingface.co/datasets/google/polaris-bench"><img src="https://img.shields.io/badge/Data-CC--BY--4.0-lightgrey.svg" alt="Data License"></a>
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=1PT3EQoAAAAJ">Xia Hu</a><sup>1</sup>, 
  <a href="https://scholar.google.com/citations?user=d901d8AAAAAJ">Zhenrui Yue</a><sup>1</sup>, 
  <a href="https://scholar.google.com/citations?user=pG-Kj_IAAAAJ">Brian Potetz</a><sup>1</sup>, 
  <a href="https://scholar.google.com/citations?user=c_3w1c4AAAAJ">Howard Zhou</a><sup>1</sup>, 
  <a href="https://scholar.google.com/citations?user=K406kGgAAAAJ">Leonidas Guibas</a><sup>1,2</sup>, 
  <a href="https://scholar.google.com/citations?user=xN1T12AAAAAJ">Chun-Ta Lu</a><sup>3</sup>, 
  <a href="https://scholar.google.com/citations?user=G4581w0AAAAJ">Zhicheng Wang</a><sup>1</sup>
  <br>
  <sup>1</sup>Google DeepMind &nbsp; <sup>2</sup>Stanford University &nbsp; <sup>3</sup>Google Research
</p>

---

## Highlights

| | |
|---|---|
| 🧩 **10,800** evaluation problems | 🧭 **4** coordinate systems (Cartesian, Polar, Hexagonal, Octagonal) |
| 📐 **53** visual reasoning tasks across 5 categories | 📉 **~32 pt** average Cartesian to Polar accuracy drop |
| 🏷️ **14** state-of-the-art MLLMs evaluated | 🔬 Paired Cartesian-Polar controlled evaluation |

## Overview

Current Multimodal LLMs achieve strong scores on visual reasoning benchmarks, but do these scores reflect genuine visual understanding? We identify a pervasive vulnerability: **the Cartesian Shortcut**.

> Models systematically discretize orthogonal grid-based layouts into explicit textual coordinates, offloading visual reasoning onto text-based deduction. This inflates performance on standard benchmarks.

**Polaris-Bench** dismantles this shortcut by re-formulating 53 visual reasoning tasks in **Polar coordinate space**, paired with Cartesian counterparts under identical logical constraints. Results show that frontier models achieving 70–83% on Cartesian layouts **collapse to 31–39%** on Polar equivalents.

<p align="center">
  <img src="docs/static/figures/polar_figure_representative_example.png" width="95%" alt="Representative task pairs in Polaris-Bench across five cognitive categories">
</p>

## Leaderboard

All models evaluated under high reasoning mode. Sorted by **Polar accuracy (P)**. Full per-category results on the [project page](https://google-deepmind.github.io/polaris-bench).

| # | Model | Type | Cartesian (%) | Polar (%) | Drop (Δ) |
|:-:|-------|:----:|:---:|:---:|:---:|
| 👤 | **Human** | Baseline | **94.5** | **88.8** | **-5.7** |
| 1 | **GPT-5.2** | Closed | 77.4 | **39.2** | -38.2 |
| 2 | **Gemini-3.1-Pro** | Closed | **82.6** | 35.9 | -46.7 |
| 3 | **Qwen3.5-397B-A17B** | Open | 72.9 | 35.0 | -37.9 |
| 4 | **Gemini-3-Flash** | Closed | 71.0 | 33.8 | -37.2 |
| 5 | **Kimi-k2.5** | Open | 69.0 | 31.1 | -37.8 |
| 6 | **Gemma-4-31B** | Open | 60.5 | 31.0 | -29.5 |
| 7 | **Claude-Sonnet-4.6** | Closed | 44.4 | 25.9 | -18.5 |
| 8 | **Gemini-2.5-Pro** | Closed | 38.4 | 25.3 | -13.2 |
| 9 | **Gemini-3-Flash-lite** | Closed | 46.8 | 24.6 | -22.2 |
| 10 | **Gemma-4-26B** | Open | 47.2 | 22.9 | -24.4 |
| 11 | **Grok-4-Fast-Reasoning** | Closed | 31.0 | 22.3 | -8.7 |
| 12 | **Grok-4-0709** | Closed | 33.0 | 21.8 | -11.2 |
| 13 | **Gemini-2.5-Flash** | Closed | 32.2 | 21.1 | -11.2 |
| 14 | **Mistral-Small-2503** | Open | 19.4 | 19.0 | -0.4 |
| - | **Random Baseline** | Baseline | 15.8 | 15.8 | 0.0 |

## Installation

```bash
git clone https://github.com/google-deepmind/polaris-bench.git
cd polaris-bench
pip install -r requirements.txt
```

## Quick Start

You can inspect the benchmark using either the HuggingFace Hub dataset or the local offline sample data:

```python
from evaluation.data_loader import PolarisDataLoader

# 1. Load from local file or HuggingFace Hub ('google/polaris-bench')
loader = PolarisDataLoader("examples/sample_tasks/sample_tasks.json")

# 2. Filter by task and coordinate system
cart_examples = loader.get_dataset(task="sudoku", question_type="cartesian")
print(f"Loaded {len(cart_examples)} Cartesian Sudoku tasks.")

# 3. Retrieve paired instances (Cartesian vs. Polar)
pairs = loader.get_paired_dataset(task="sudoku")
cart_sample, polar_sample = pairs[0]

print("Cartesian Image:", cart_sample["image_path"])
print("Polar Image:    ", polar_sample["image_path"])
print("Ground Truth:   ", cart_sample["answer"])
```

> **Offline sample data**: A standalone set of 20 representative paired tasks is provided in [`examples/sample_tasks/`](examples/sample_tasks/) for quick local inspection without downloading the full dataset.

## Evaluation

Run the evaluation script against your model's predictions:

```bash
python -m evaluation.evaluate \
  --predictions predictions.json \
  --ground-truth polaris_bench.json
```

**Supported Prediction Formats**:

1. **3-part key** (`task::index::question_type`):
```json
{
  "sudoku::example_001::cartesian": "D",
  "sudoku::example_001::polar": "B",
  "maze::example_042::polar": "A"
}
```

2. **Nested by coordinate system**:
```json
{
  "cartesian": {
    "sudoku::example_001": "D"
  },
  "polar": {
    "sudoku::example_001": "B"
  }
}
```

3. **List of prediction records**:
```json
[
  {
    "task": "sudoku",
    "index": "example_001",
    "question_type": "polar",
    "prediction": "B"
  }
]
```

To evaluate only a specific coordinate system, pass `--question-type`:
```bash
python -m evaluation.evaluate \
  --predictions predictions.json \
  --question-type polar
```

The script automatically formats a terminal summary table and exports a detailed JSON report:
```
==================================================================================
                         POLARIS-BENCH EVALUATION REPORT
==================================================================================
Total Evaluated: 10800 | Overall Accuracy: 59.2%
Cartesian Acc:   77.4% | Polar Acc: 39.2% | Drop (Cartesian - Polar): 38.2 pt
----------------------------------------------------------------------------------
Category                             | Cartesian (%) | Polar (%)  | Drop (pt) 
----------------------------------------------------------------------------------
Algorithmic Logic & Simulation       | 78.5          | 41.2       | 37.3      
Combinatorics & Probability          | 74.2          | 36.8       | 37.4      
Navigation & Routing                 | 81.0          | 40.5       | 40.5      
Spatial Transformation & Geometry    | 76.1          | 38.9       | 37.2      
Visual Pattern Matching              | 77.2          | 38.6       | 38.6      
==================================================================================
```

## Dataset Structure

Each record contains 6 fields (+ `image` in the HuggingFace Parquet version):

| Field | Type | Description |
|-------|------|-------------|
| `task` | string | Task name (e.g., `sudoku`, `maze`, `shape_fitting`) |
| `index` | string | Sample index (e.g., `example_001`) |
| `question_type` | string | `cartesian`, `polar`, `hexagonal`, or `octagonal` |
| `image` | PIL.Image | Evaluation image (RGBA, ~3000×3000 to 4000×3600) |
| `image_path` | string | Relative path (e.g., `images/sudoku/sudoku_example_001_cartesian.png`) |
| `question` | string | Evaluation question text |
| `answer` | string | Ground-truth answer (option labels, digits, coordinates, strings, or lists) |

## Task Taxonomy

53 tasks organized into 5 cognitive categories:

| Category | Tasks | Count |
|----------|-------|:-----:|
| **Visual Pattern Matching** | pattern_completion, pattern_prediction, layer_completion, shape_completion, shape_fitting, jigsaw_matching, odd_piece_out, fragment_matching, anomaly_detection, template_matching, impossible_shape | 11 |
| **Spatial Transformation & Geometry** | grid_rotation, pivot_rotation, mirror_reflection, grid_folding, rotation_center, rotation_matching, area_balancing, minimum_flips, wall_follower, letter_collection, turn_counting, pipe_lengths, word_search | 13 |
| **Navigation & Routing** | maze, shortest_path, bounded_path_finding, wrapping_path_finding, bounded_diagonal_paths, wrapping_diagonal_paths, bounded_knight_paths, knight_paths, checkpoint_paths, monotonic_path, rule_based_navigation, absolute_navigation, egocentric_navigation, wrapping_navigation | 14 |
| **Combinatorics & Probability** | path_counting, lattice_paths, area_counting, edge_counting, uncut_cells, maximum_collection, longest_path, largest_number_path, curve_length | 9 |
| **Algorithmic Logic & Simulation** | sudoku, four_color, n_queens, random_walk, collision_detection, bouncing_point | 6 |

## Citation

```bibtex
@misc{hu2026cartesianshortcutreevaluatevision,
  title         = {The Cartesian Shortcut: Re-evaluate Vision Reasoning in Polar Coordinate Space},
  author        = {Xia Hu and Zhenrui Yue and Brian Potetz and Howard Zhou and Leonidas Guibas and Chun-Ta Lu and Zhicheng Wang},
  year          = {2026},
  eprint        = {2605.09883},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.09883}
}
```

## License

- **Code**: [Apache License 2.0](LICENSE)
- **Dataset**: [Creative Commons Attribution 4.0 (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)

## Acknowledgments

Polaris-Bench draws inspiration from and builds upon prior work in visual reasoning evaluation, including [BabyVision](https://arxiv.org/abs/2601.06521), [EMMA-Bench](https://openreview.net/forum?id=v26vwjxOEz), [MathVista](https://mathvista.github.io/), [MEGABench](https://megabench.github.io/), [OmniSpatial](https://openreview.net/forum?id=6nZKT2rL0H), [VGRP-Bench](https://arxiv.org/abs/2503.23064), and [GRASP](https://arxiv.org/abs/2407.01892), among others.

*This is not an officially supported Google product.*

