# Polaris-Bench Sample Tasks

This directory contains representative paired evaluation instances from the **Polaris-Bench** benchmark, showcasing paired **Cartesian** and **Polar** coordinate representations under identical logical constraints.

## Directory Structure

```
examples/sample_tasks/
├── sample_tasks.json          # Paired evaluation instances with questions, answers, and image paths
├── README.md                  # This documentation
└── images/                    # Paired PNG images organized by task
    ├── sudoku/
    │   ├── sudoku_sudoku_001_cartesian.png
    │   └── sudoku_sudoku_001_polar.png
    ├── maze/
    │   ├── maze_maze_001_cartesian.png
    │   └── maze_maze_001_polar.png
    └── ... (20 representative tasks across 5 categories)
```

## Quick Inspection (Python)

### Option 1: Using PolarisDataLoader (Recommended)

```python
from evaluation.data_loader import PolarisDataLoader

loader = PolarisDataLoader("examples/sample_tasks/sample_tasks.json")
pairs = loader.get_paired_dataset()
print(f"Loaded {len(pairs)} representative paired tasks.")

for cart, polar in pairs[:3]:
    print(f"\nTask: {cart['task']} ({cart.get('category')}) | Index: {cart['index']}")
    print(f"  Cartesian Image: {cart['image_path']}")
    print(f"  Polar Image:     {polar['image_path']}")
    print(f"  Ground Truth:    {cart['answer']}")
```

### Option 2: Using Raw JSON

```python
import json

with open("examples/sample_tasks/sample_tasks.json") as f:
    samples = json.load(f)

print(f"Loaded {len(samples)} representative tasks.")

for s in samples[:3]:
    print(f"\nTask: {s['name']} ({s['category_name']}) | Index: {s['index']}")
    print(f"  Cartesian Image: {s['cartesian']['gh_image']}")
    print(f"  Polar Image:     {s['polar']['gh_image']}")
    print(f"  Ground Truth:    {s['answer']}")
```
