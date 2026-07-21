# Reproducing the tracked relation-model pipeline

This document covers the code currently tracked in this repository: parsing the 4,694-equation catalog, constructing the implication graph, and training the initial neural relation model described in the accompanying post.

The later clustering, syntax-rule, and evaluation experiments reported in the post are not fully reproducible from the tracked files alone. Their scripts and generated artifacts are not currently part of the public repository.

## Requirements

- Python 3.10 or newer
- [PyTorch](https://pytorch.org/get-started/locally/)
- Enough free disk space for the generated implication graph (approximately 340 MB)

Create an isolated environment and install PyTorch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
```

For CUDA or another accelerated backend, use the platform-specific installation command from the PyTorch documentation instead.

## 1. Convert the equation catalog to postfix notation

```bash
python convert_equations_to_postfix.py
```

This reads `equations.txt` and writes `equations_postfix.txt`. A successful run reports:

```text
Converted equations: 4694
Wrote postfix equations: equations_postfix.txt
```

Custom paths can be supplied with `--input` and `--output`.

## 2. Construct the full implication graph

```bash
python construct_full_implication_graph.py
```

This reads `explicit_implications.json` and `equations_postfix.txt`, computes the non-reflexive transitive closure, and writes `implication_graph.json`.

For the tracked inputs, the source contains 10,667 explicit implications and the generated graph contains 8,173,585 implication edges. The resulting JSON file is approximately 340 MB.

Custom paths can be supplied with `--input`, `--equations`, and `--output`.

## 3. Train the relation model

A small smoke run is useful before starting the default training job:

```bash
python train_relation_model.py \
  --epochs 1 \
  --samples-per-epoch 256 \
  --validation-samples 64 \
  --batch-size 32 \
  --no-save
```

Run the default training configuration with:

```bash
python train_relation_model.py
```

The trainer automatically selects CUDA, MPS, or CPU. Override this with `--device`, for example `--device cpu` or `--device cuda`.

Unless `--no-save` is used, the trained checkpoint is written to `relation_model.pt`. Use `python train_relation_model.py --help` for model dimensions, sample counts, optimization settings, input paths, and checkpoint options.

## Generated files

| File | Produced by | Typical size |
| --- | --- | ---: |
| `equations_postfix.txt` | `convert_equations_to_postfix.py` | under 1 MB |
| `implication_graph.json` | `construct_full_implication_graph.py` | about 340 MB |
| `relation_model.pt` | `train_relation_model.py` | depends on model dimensions |

The training code intentionally processes equations one at a time and is designed for clarity rather than maximum throughput.
