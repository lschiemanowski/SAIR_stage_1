# SAIR Stage 1 Relation Model

This is the companion repository for the blogpost
[The SAIR distillation challenge stage 1](https://loss.md/posts/2026-05-04-sair-distillation-challenge-stage-1-postmortem/index.html).
It contains the cleaned-up code and reproduction workflow for the
relation-model tables and figures in the post. For motivation, interpretation,
and details of the competition result, see the blogpost.

## Dependencies

The scripts use PyTorch for training, scikit-learn for k-means clustering,
matplotlib for the figures, and Hugging Face `datasets` for fetching the public
SAIR problem splits:

```bash
python3 -m pip install torch numpy scikit-learn matplotlib tqdm datasets
```

Use the PyTorch install command appropriate for your CUDA setup if you want GPU
training.

## Recovering the Blogpost Results

The commands below recreate the relation-model artifacts used for the blogpost:
the trained model, the relation-fingerprint clusters, the `8 x 8` and collapsed
`4 x 4` block oracles, the public/released SAIR split evaluations, and the PCA
figures. Training and k-means are seeded, but exact floating-point values can
still vary slightly across hardware and PyTorch versions.

The official leaderboard numbers in the post come from the SAIR scoring system,
not from this repository. The order-5 leaderboard result is also not reproduced
here: the relation-cluster oracle below is for the 4694 order-at-most-4 equations.
The `Raw Test` row in the post came from an earlier random held-out split of
the full implication graph; this repo instead reports full-graph oracle metrics
in the clustering and collapse reports.

### 1. Prepare the Implication Graph

The repo ships with infix equations and explicit implications. The training
scripts expect postfix equations and the full implication closure:

```bash
python3 convert_equations_to_postfix.py
python3 construct_full_implication_graph.py
```

This creates:

- `equations_postfix.txt`
- `implication_graph.json`

### 2. Train the Relation Model

A small smoke run:

```bash
python3 train_relation_model.py \
  --epochs 1 \
  --samples-per-epoch 8192 \
  --validation-samples 2048 \
  --batch-size 256 \
  --device auto
```

The larger run used for the blogpost-style artifacts:

```bash
python3 train_relation_model.py \
  --checkpoint relation_model.pt \
  --epochs 10 \
  --samples-per-epoch 2000000 \
  --validation-samples 200000 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --weight-decay 1e-5 \
  --feature-dim 256 \
  --relation-hidden-dim 128 \
  --relation-bottleneck-dim 16 \
  --seed 0 \
  --device auto
```

Training uses tqdm progress bars when `tqdm` is installed. Pass
`--no-progress` to disable them.

### 3. Cluster Relation Fingerprints

Cluster equations by their learned relation fingerprints. The blogpost uses the
`k=8` clustering for the main table and also evaluates finer `k=16`, `k=32`,
and `k=64` cluster oracles as diagnostics:

```bash
for K in 8 16 32 64; do
  python3 cluster_relation_fingerprints.py \
    --checkpoint relation_model.pt \
    --num-clusters "$K" \
    --output-dir "relation_fingerprint_runs/k${K}" \
    --save-fingerprints \
    --seed 0 \
    --n-init 20 \
    --device auto
done
```

The script writes `summary.json`, `report.md`, and, with `--save-fingerprints`,
`fingerprints.npy` under each run directory. The induced
`k x k` oracle records, for each source/target cluster pair, the empirical
implication rate in the full graph.

### 4. Collapse the Learned 8 Clusters to 4 Types

The blogpost uses the explicit collapse

```bash
python3 collapse_cluster_oracle.py \
  --summary relation_fingerprint_runs/k8/summary.json \
  --mapping 'S:1,3,6,7,8;2:2;4:4;5:5'
```

This writes:

- `relation_fingerprint_runs/k8/collapsed_k4_summary.json`
- `relation_fingerprint_runs/k8/collapsed_k4_summary.md`

You can also ask the script to find a greedy four-cluster collapse:

```bash
python3 collapse_cluster_oracle.py \
  --summary relation_fingerprint_runs/k8/summary.json \
  --target-clusters 4 \
  --output-json relation_fingerprint_runs/k8/greedy_k4_summary.json
```

### 5. Fetch the Public SAIR Splits

The public and released evaluation splits are hosted as the Hugging Face dataset
[`SAIRfoundation/equational-theories-selected-problems`](https://huggingface.co/datasets/SAIRfoundation/equational-theories-selected-problems).
Save the order-at-most-4 splits as local JSONL files:

```bash
python3 - <<'PY'
import json
from pathlib import Path

from datasets import load_dataset

dataset = "SAIRfoundation/equational-theories-selected-problems"
subsets = [
    "normal",
    "hard",
    "hard1",
    "hard2",
    "hard3",
    "evaluation_normal",
    "evaluation_hard",
    "evaluation_extra_hard",
]

output_dir = Path("SAIR_selected_problems")
output_dir.mkdir(exist_ok=True)

for subset in subsets:
    rows = load_dataset(dataset, subset, split="train")
    path = output_dir / f"{subset}.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row)) + "\n")
    print(f"wrote {path}")
PY
```

### 6. Evaluate the Oracles on the Splits

This command evaluates the learned `8`-cluster oracle, the collapsed `4`-type
oracle, the finer diagnostic clusterings, and the hand-written syntax oracle on
the same split files:

```bash
python3 reproduce_blogpost_tables.py \
  --oracle learned8=relation_fingerprint_runs/k8/summary.json \
  --oracle collapse4=relation_fingerprint_runs/k8/collapsed_k4_summary.json \
  --oracle clusters16=relation_fingerprint_runs/k16/summary.json \
  --oracle clusters32=relation_fingerprint_runs/k32/summary.json \
  --oracle clusters64=relation_fingerprint_runs/k64/summary.json \
  --syntax \
  --splits \
    SAIR_selected_problems/normal.jsonl \
    SAIR_selected_problems/hard.jsonl \
    SAIR_selected_problems/hard1.jsonl \
    SAIR_selected_problems/hard2.jsonl \
    SAIR_selected_problems/hard3.jsonl \
    SAIR_selected_problems/evaluation_normal.jsonl \
    SAIR_selected_problems/evaluation_hard.jsonl \
    SAIR_selected_problems/evaluation_extra_hard.jsonl \
  --output-md blogpost_oracle_tables.md \
  --output-json blogpost_oracle_tables.json \
  --top-error-blocks 12
```

`equation_type_heuristics.py` contains the hand-written `S / 2 / 4 / 5`
syntax oracle. `reproduce_blogpost_tables.py` is a thin wrapper for producing a
single Markdown table from several oracle artifacts and split directories.

### 7. Generate the PCA Figures

Generate the PCA views used in the blogpost. These keep the same two
coordinates fixed and recolor the equations by learned `8`-cluster labels, the
`4`-type coarsening, and the hand-written syntax rules:

```bash
python3 plot_cluster_pca_views.py \
  --summary relation_fingerprint_runs/k8/summary.json \
  --fingerprints relation_fingerprint_runs/k8/fingerprints.npy \
  --output-prefix blogpost_cluster_pca \
  --markdown-snippet blogpost_cluster_pca_figures.md
```

The generated Markdown tables and figures are the artifacts to compare with the
tables and images in the blogpost.
