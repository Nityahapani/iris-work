# Temporal Graph Sparsification through Edge Retirement Scheduling

**Author:** Nitya Hapani  
**Competition:** IRIS / ISEF — Software Design (Systems Software)  
**Sponsor Category:** Microsoft  

---

## Overview

Standard graph sparsification asks *which* edges to keep. This project asks a fundamentally different question: **when has each edge fulfilled its purpose?**

We introduce **Edge Retirement Scheduling** — a temporal optimization framework in which every edge in a Graph Neural Network (GNN) is assigned an optimal retirement time during training, rather than being permanently retained or removed upfront. Dense connectivity is exploited early in training when long-range propagation helps establish representations; edges are gradually retired once their contribution to the learned embeddings falls below a provable threshold.

This transforms sparsification from a static graph compression problem into a **temporal learning process**, with theoretical guarantees on representation preservation, prediction distortion, and computational savings.

---

## Key Contributions

| Contribution | Description |
|---|---|
| **Edge Retirement Schedule** | A function τ(e) → {0,...,T,+∞} assigning each edge an optimal retirement time |
| **Jacobian Influence Estimator** | Efficient gradient-based approximation of per-edge representation influence Ie(t) |
| **Safe Retirement Criterion** | Provable ε-bound on representation and prediction distortion at retirement |
| **Temporal Objective** | Formal optimization over schedules; strict generalization of all static methods |
| **Adaptive Scheduler** | Cooling schedule + batch retirement with cumulative distortion control |

---

## Theoretical Guarantees (summary)

- Single edge retirement: `‖H_t - H_t^{-e}‖_F ≤ ε`  
- Prediction distortion: `‖Ŷ_t - Ŷ_t^{-e}‖_F ≤ K_f · ε`  
- k simultaneous retirements: cumulative bound `kε`  
- Temporal schedules strictly dominate all static sparsification methods  
- Redundant edges guaranteed to retire in finite steps under convergence  

See [`docs/theory.md`](docs/theory.md) for the full proof document.

---

## Project Structure

```
iris-work/
├── tgs/                    # Core library
│   ├── core/               # Temporal graph, edge set management
│   ├── models/             # GCN + generalized GNN wrappers
│   ├── schedulers/         # Retirement schedulers (adaptive, fixed-ε)
│   ├── evaluation/         # Metrics: accuracy, FLOPs, memory, sparsity
│   └── utils/              # Logging, config, reproducibility
├── experiments/            # Experiment scripts (Cora, CiteSeer, OGB, ablations)
├── configs/                # YAML configs for each experiment
├── scripts/                # Training entry points
├── tests/                  # Unit tests for core components
└── docs/                   # Theory document, figures, writeup notes
```

---

## Datasets

| Dataset | Nodes | Edges | Task |
|---|---|---|---|
| Cora | 2,708 | 5,429 | Node classification |
| CiteSeer | 3,327 | 4,732 | Node classification |
| PubMed | 19,717 | 44,338 | Node classification |
| OGB-arxiv | 169,343 | 1,166,243 | Node classification |

---

## Baselines

- Random sparsification (static)
- Effective Resistance sampling
- Local Degree pruning
- UGS (task-aware GNN sparsification)
- Dense training (upper bound)

---

## Quickstart

```bash
pip install -r requirements.txt

# Train with edge retirement on Cora
python scripts/train.py --config configs/cora_gcn.yaml

# Run full evaluation suite
python scripts/evaluate.py --config configs/eval_full.yaml

# Ablation: influence approximation method
python experiments/ablation_influence.py
```

---

## Implementation Log

All development is committed incrementally. The git log serves as a research diary — each commit message describes what was built and why.

---

## Citation

```
Hapani, N. (2026). Temporal Graph Sparsification through Edge Retirement Scheduling.
IRIS Science and Engineering Fair.
```
