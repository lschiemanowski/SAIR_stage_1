# SAIR Stage 1 Relation Model

This repository contains a pedagogical version of the relation-model workflow
used for SAIR Stage 1 implication prediction.

The task is binary implication prediction:

```text
equation E_left  ->  equation E_right
```

The model has three inspectable stages:

1. `postfix_equation_tensor.py` compiles each postfix equation into a flat
   syntax table.
2. `compiled_equation_encoder.py` batches those tables and maps each equation
   index to a learned equation embedding.
3. `relation_head.py` scores ordered pairs of equation embeddings with the
   feature vector

   ```text
   [e_left ; e_right ; e_left - e_right ; e_left * e_right].
   ```

The equation encoder uses the symmetric aggregation

```text
[h_left + h_right ; |h_left - h_right|] -> MLP.
```

## Data Preparation

The repo ships with infix equations and explicit implications. The training
scripts expect postfix equations and the full implication closure:

```bash
python3 convert_equations_to_postfix.py
python3 construct_full_implication_graph.py
```

This creates:

- `equations_postfix.txt`
- `implication_graph.json`

## Training

A small smoke run:

```bash
python3 train_relation_model.py \
  --epochs 1 \
  --samples-per-epoch 8192 \
  --validation-samples 2048 \
  --batch-size 256 \
  --device auto
```

A larger run:

```bash
python3 train_relation_model.py \
  --epochs 10 \
  --samples-per-epoch 2000000 \
  --validation-samples 200000 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --weight-decay 1e-5 \
  --feature-dim 256 \
  --relation-hidden-dim 128 \
  --relation-bottleneck-dim 16 \
  --device auto
```

Training uses tqdm progress bars when `tqdm` is installed. Pass
`--no-progress` to disable them.

## Relation Fingerprint Clustering

Once a checkpoint exists, cluster equations by their learned relation
fingerprints:

```bash
python3 cluster_relation_fingerprints.py \
  --checkpoint relation_model.pt \
  --num-clusters 8 \
  --save-fingerprints \
  --device auto
```

The script writes `summary.json`, `report.md`, and, with `--save-fingerprints`,
`fingerprints.npy` under `relation_fingerprint_runs/<timestamp>/`. The induced
`k x k` oracle records, for each source/target cluster pair, the empirical
implication rate in the full graph.

## Blogpost Oracle Analysis

The root-level analysis scripts are intentionally small and composable.

Collapse a learned `8 x 8` oracle to four types:

```bash
python3 collapse_cluster_oracle.py \
  --summary relation_fingerprint_runs/<run>/summary.json \
  --target-clusters 4
```

To reproduce the manual collapse discussed in the post, pass the mapping
explicitly:

```bash
python3 collapse_cluster_oracle.py \
  --summary relation_fingerprint_runs/<run>/summary.json \
  --mapping 'S:1,3,6,7,8;2:2;4:4;5:5'
```

Evaluate learned-cluster or syntax oracles on SAIR-style JSONL splits:

```bash
python3 evaluate_oracle_on_splits.py \
  --oracle-summary collapsed=relation_fingerprint_runs/<run>/collapsed_k4_summary.json \
  --syntax \
  --splits SAIR_challenge_dataset/data \
  --output-md oracle_eval.md
```

`equation_type_heuristics.py` contains the hand-written `S / 2 / 4 / 5`
syntax oracle. `reproduce_blogpost_tables.py` is a thin wrapper for producing a
single Markdown table from several oracle artifacts and split directories.

Generate the PCA views used in the blogpost. These keep the same two
coordinates fixed and recolor the equations by learned `8`-cluster labels, the
`4`-type coarsening, and the hand-written syntax rules:

```bash
python3 plot_cluster_pca_views.py \
  --summary relation_fingerprint_runs/<run>/summary.json \
  --fingerprints relation_fingerprint_runs/<run>/fingerprints.npy \
  --output-prefix blogpost_cluster_pca \
  --markdown-snippet blogpost_cluster_pca_figures.md
```
