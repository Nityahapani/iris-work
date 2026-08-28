# Temporal Graph Sparsification (TGS)

**Author:** Nitya Hapani  
**Competition:** IRIS / ISEF — Software Design (Systems Software)  
**Sponsor Category:** Microsoft  

> **Sparsify when edges become redundant — not before the model has learned what they mean.**

Temporal Graph Sparsification (**TGS**) is a training-time graph sparsification framework for Graph Neural Networks (GNNs). Instead of deciding which edges to remove before training, TGS starts from the dense graph, allows node representations to mature, and progressively retires edges during training.

The central empirical result is simple:

**The timing of sparsification is a causal part of the problem.**

Across controlled interventions, matched-sparsity comparisons, prospective validation, timing sweeps, factorial experiments, and multi-seed real-graph evaluations, the evidence consistently shows that delaying edge retirement can preserve representation quality while substantially reducing the final graph.

---

## Table of Contents

- [Why TGS?](#why-tgs)
- [The Core Idea](#the-core-idea)
- [What the Evidence Says](#what-the-evidence-says)
- [The Complete Causal Chain](#the-complete-causal-chain)
- [Structural Predictability](#structural-predictability)
- [Prospective Validation](#prospective-validation)
- [Timing Is the Mechanism](#timing-is-the-mechanism)
- [Maturation Predicts When to Prune](#maturation-predicts-when-to-prune)
- [TGS Algorithm](#tgs-algorithm)
- [Influence Estimator](#influence-estimator)
- [Scheduler](#scheduler)
- [JSD Disagreement Signal](#jsd-disagreement-signal)
- [Benchmark Results](#benchmark-results)
- [Real-World Graph Results](#real-world-graph-results)
- [Architecture Generalization](#architecture-generalization)
- [Ablations](#ablations)
- [Runtime and Memory](#runtime-and-memory)
- [What TGS Is and Is Not](#what-tgs-is-and-is-not)
- [Recommended Configuration](#recommended-configuration)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Limitations and Caveats](#limitations-and-caveats)
- [Project Structure](#project-structure)
- [Conclusion](#conclusion)

---

## Why TGS?

GNNs perform message passing over graph edges. In real graphs, however, the full edge set can contain substantial redundancy.

This creates a fundamental tradeoff:

- **Keep every edge:** preserve information, but pay the full computation and memory cost.
- **Prune before training:** reduce cost, but risk removing edges whose usefulness has not yet been revealed.
- **Prune during training:** allow the model to learn first, then remove edges as their contribution becomes identifiable.

Most conventional sparsification methods make their topology decision **before representation learning**. Once an edge is removed, the model never gets to learn whether that edge was useful.

TGS changes the order of operations.

### Static sparsification

```text
Initial graph
     │
     ▼
Choose edges to keep
     │
     ▼
Train GNN on sparse graph
```

### Temporal Graph Sparsification

```text
Dense graph
     │
     ▼
Representation learning / warmup
     │
     ▼
Representations mature
     │
     ▼
Estimate edge redundancy
     │
     ▼
Retire edges progressively
     │
     ▼
Train on increasingly sparse graph
```

The hypothesis is not merely that TGS finds a better final topology.

It is that **the model needs a temporal window in which edges can participate before their redundancy becomes safely identifiable.**

---

# The Core Idea

TGS separates two decisions that static pruning conflates:

1. **Which edges are eventually redundant?**
2. **When is it safe to remove them?**

The experiments strongly support the second question as the causal ingredient.

A particularly decisive intervention keeps the topology fixed and changes only the retirement time:

| Topology | Retire at t=0 | Retire at t=20 |
|---|---:|---:|
| Static ER | 0.764 | 0.910 |
| Oracle/TGS topology | 0.750 | 0.910 |
| Dense | — | 0.910 |

The topology at `t=0` can be *better informed* and still perform substantially worse than the same topology retired after representation learning.

**Oracle t=0 → t=20: +16.0 percentage points.**

This is the cleanest evidence that topology quality alone cannot explain the TGS effect.

---

# What the Evidence Says

The TGS results form a single experimentally connected story:

| Causal link | Evidence | Strength |
|---|---|---|
| Graph structure → TGS utility | 224 prospective graphs; MCC `+0.829` | Very strong |
| `H × CV` → TGS utility | 180 prospective evaluations; MCC `+0.87–+0.94` | Very strong |
| `H × CV` → representation maturation | Probe@20 correlation `r=+0.944` | Very strong |
| Representation maturation → `τ_G` | Spearman `ρ=-0.728`, `p=0.041` | Significant |
| `τ_G` → recovery after pruning | Oracle timing: `+16pp` from `t=0 → t=20` | Exact intervention |
| Better topology → better outcome | Falsified by oracle timing | Topology alone insufficient |
| Timing → outcome | Random/Reverse/TGS temporal ≈ each other, all beat static | Strong causal evidence |
| Timing + structural regime → TGS advantage | Wisconsin factorial interaction `β_int=-0.067` | Strong intervention evidence |

The resulting mechanism is:

```text
Graph structure
     │
     ▼
Representation maturation speed
     │
     ▼
Task-relevant information becomes encoded
     │
     ▼
Previously useful edges become redundant
     │
     ▼
Those edges can now be retired safely
     │
     ▼
Sparse graph + preserved accuracy
```

---

# The Complete Causal Chain

## 1. Structure predicts whether temporal sparsification is useful

A frozen structural score was defined as:

```text
SCORE(G) = homophily(G) × deg_cv(G)
```

with the prospective decision rule:

```text
SCORE(G) > 0.0505
```

The threshold was frozen after calibration and was not modified during the later validation sets.

Across the combined independent evaluation:

- **224 graphs**
- Accuracy: **91.1%**
- Balanced accuracy: **90.9%**
- MCC: **+0.829**
- TP: 112
- FP: 18
- FN: 2
- TN: 92
- Pearson `r = +0.725`
- `p = 7.3 × 10⁻³⁸`

A large-scale prospective validation independently produced:

- **150 graphs**
- Accuracy: **93.3%**
- Balanced accuracy: **93.2%**
- MCC: **+0.869**
- Precision: **90.4%**
- Recall: **97.4%**
- Specificity: **89.0%**
- Score → TGS-vs-dense delta: `r=+0.755`, `p=6.27×10⁻²⁹`

Rule-positive graphs had mean improvement of **+7.64pp**, while rule-negative graphs averaged only **+0.24pp**.

---

## 2. The product structure is experimentally supported

A controlled `H × CV` factorial grid tested whether homophily and degree variation interact rather than merely contribute independently.

The regression was:

```text
Δ = β₀ + β₁H + β₂CV + β₃(H×CV)
```

Results:

| Term | Estimate |
|---|---:|
| `β₀` | -0.0068 |
| `β₁` | +0.151 |
| `β₂` | +0.149 |
| `β₃` | +0.089 |

Model:

- Full `R² = 0.714`
- Without interaction `R² = 0.634`
- `ΔR² = +0.080`
- Interaction permutation `p = 0.006`

Quadrant means:

| Regime | Mean Δ |
|---|---:|
| Low H × Low CV | +3.8pp |
| Low H × High CV | +5.4pp |
| High H × Low CV | +9.3pp |
| **High H × High CV** | **+17.8pp** |

The advantage is concentrated where the interaction predicts it should be.

---

## 3. Structure predicts representation maturation

The `SCORE(G)` is not merely correlated with the final TGS gain.

It also predicts how quickly useful representations emerge.

At epoch 20, linear-probe accuracy correlated with score at:

```text
r = +0.944
```

The direct maturation experiment on Cora produced:

| Warmup | Probe | TGS | Static | TGS − Static |
|---:|---:|---:|---:|---:|
| 0 | 0.382 | 0.751 | 0.726 | +2.5pp |
| 10 | 0.715 | 0.773 | 0.726 | +4.7pp |
| 20 | 0.805 | 0.795 | 0.726 | +6.9pp |
| 30 | 0.822 | 0.798 | 0.726 | +7.2pp |
| **40** | **0.821** | **0.801** | **0.726** | **+7.5pp** |
| 60 | 0.818 | 0.801 | 0.726 | +7.5pp |
| 80 | 0.803 | 0.801 | 0.726 | +7.5pp |
| 120 | 0.793 | 0.801 | 0.726 | +7.5pp |

Probe accuracy vs TGS gain:

```text
Pearson r = +0.9448
p = 0.0004
```

This is important because the probe measures representation quality **before pruning** and does not use the test set.

---

# Structural Predictability

The early validation established several candidate predictors.

### Feature ablation tournament

| Predictor | MCC |
|---|---:|
| `H × CV` | **+0.935** |
| Homophily | **+0.935** |
| Degree CV | **+0.935** |
| H + CV | **+0.935** |
| Spectral gap | +0.874 |
| Assortativity | +0.816 |
| Random forest on 9 features | +0.655 |
| Clustering coefficient | +0.447 |
| Density | +0.236 |
| Mean degree | +0.236 |

The striking result is that a simple interpretable structural statistic matches or exceeds the black-box model.

The product `H×CV` is the most compact expression, although in the initial tournament each component alone captured the full binary signal.

---

## Score predicts the optimal retention level

The same score also predicts how aggressively TGS can sparsify.

Linear fit:

```text
critical_retention = -1.72 × SCORE + 1.08
```

with:

```text
R² = 0.853
```

Examples:

| SCORE | Approx. required retention |
|---:|---:|
| 0.15 | ~82% |
| 0.48 | ~25% |

At 25% retention:

- High-score graphs: **+12–38pp** over static
- Low-score graphs: **<5pp** improvement at any tested retention level

Thus the score is not only a classifier of *whether* TGS is useful; it also carries information about *how aggressively* it can be applied.

---

# Prospective Validation

## Locked rule

The production-style rule was frozen as:

```text
SCORE(G) = homophily × deg_cv > 0.0505
```

It was calibrated on 20 graphs and then frozen.

### Independent validation: 74 new graphs

The 74-graph test set used seeds `111–888` with zero overlap with prior experiments.

Results:

- Accuracy: **86.5%**
- MCC: **+0.758**
- Recall: **100%**
- FN: **0**
- Precision: **78.7%**
- Pearson `r=+0.739`
- `p=5.7×10⁻¹⁴`

All 10 false positives were borderline:

```text
score = 0.051–0.064
actual Δ ≈ 0–2pp
```

### Combined independent validation

Across all three test sets:

```text
N = 224 graphs
Accuracy = 91.1%
Balanced Accuracy = 90.9%
MCC = +0.829
TP = 112
FP = 18
FN = 2
TN = 92
r = +0.725
p = 7.3×10⁻³⁸
```

This is the strongest prospective evidence in the current result set.

---

# Timing Is the Mechanism

The mechanism ablation is the most important conceptual experiment.

Six methods were evaluated at identical final sparsity:

1. Static ER
2. Static influence
3. Random temporal
4. Reverse temporal
5. TGS temporal
6. Oracle temporal

## Low-score graphs

All methods were approximately equivalent:

```text
~0.37–0.42 accuracy
```

TGS vs static ER:

```text
+0.4pp
```

## High-score graphs

| Method | Mean accuracy |
|---|---:|
| Static ER | 0.845 |
| Static Influence | 0.831 |
| Random Temporal | 0.977 |
| Reverse Temporal | 0.977 |
| TGS | 0.977 |
| Oracle Temporal | 0.977 |

Temporal methods improved over static by approximately:

```text
+13.2pp
```

The crucial observation:

> Random temporal retirement performs approximately as well as influence-ranked TGS.

That means the benefit does not require perfect edge ranking.

It points instead to **when** retirement happens as the causal ingredient.

---

# Temporal Order Ablation

Cora, matched final sparsity:

| Variant | Accuracy | vs Static |
|---|---:|---:|
| TGS order | 0.818 | +8.8pp |
| Random order | 0.819 | +8.9pp |
| Reverse order | 0.814 | +8.4pp |
| Static upfront | 0.730 | baseline |

All temporal variants beat static by approximately **8–9pp**.

Even reversing the retirement order preserves the advantage.

This isolates the temporal window itself from the precise ordering of individual edge retirement.

---

# Oracle Timing Intervention

The strongest topology-vs-timing intervention used the same graph and compared retirement times.

```text
t_retire = 0      → 0.750
t_retire >= 20    → 0.910
```

Therefore:

```text
Waiting 20 epochs = +16.0pp
```

The full oracle matrix:

| Topology | t=0 | t=20 | t=40 | t=80 | Never |
|---|---:|---:|---:|---:|---:|
| Random topology | 0.8542 | — | — | — | — |
| Static ER | 0.7639 | 0.9097 | 0.9097 | 0.9097 | — |
| Oracle/TGS | 0.7500 | 0.9097 | 0.9097 | 0.9097 | — |
| Dense | — | — | — | — | 0.9097 |

Two facts are decisive:

1. **Oracle topology at t=0 is not enough.**
2. **Oracle topology at t=20 reaches dense performance.**

```text
Oracle t=0 vs t=20: +16.0pp
Oracle t=20 = Dense: True
```

This is why TGS should be understood as a **temporal learning intervention**, not merely a better static edge-selection heuristic.

---

# Maturation Predicts When to Prune

The pre-registered maturation experiment directly tested whether representation quality predicts the safe retirement time.

Protocol:

1. Train the dense GCN.
2. Freeze representations at candidate warmup epochs.
3. Fit a linear probe without graph message passing.
4. Measure representation quality.
5. Start TGS retirement at each warmup.
6. Compare representation maturity against TGS gain.

### Cora

The TGS advantage rises as representations mature and plateaus at approximately 40 epochs.

```text
Probe gain → TGS gain
Pearson r = +0.9448
p = 0.0004
```

### Cross-dataset

| Dataset | Probe gain 0→40 | TGS gain |
|---|---:|---:|
| Cora | +0.439 | +0.075 |
| CiteSeer | +0.167 | +0.000 |
| PubMed | +0.119 | +0.009 |

Across these three points:

```text
Pearson r = +0.969
```

The direction is consistent:

> Faster representation maturation corresponds to a larger benefit from timely sparsification.

---

# SCORE → τ_G: Structure Predicts Maturation Speed

The recovery time `τ_G` was measured over graphs spanning scores from approximately 0.03 to 0.48.

| SCORE | τ_G |
|---:|---:|
| 0.030 | 200 |
| 0.053 | 20 |
| 0.146 | 20 |
| 0.193 | 50 |
| 0.245 | 20 |
| 0.311 | 10 |
| 0.400 | 20 |
| 0.482 | 10 |

Correlation:

```text
Spearman ρ = -0.728
p = 0.041
```

Pearson:

```text
r = -0.568
p = 0.141
```

The Spearman result is significant at the eight tested points, while the Pearson result is not. The rank-order relationship is therefore the stronger supported claim.

The observed mechanism is:

```text
Higher SCORE
     ↓
Faster representation maturation
     ↓
Smaller τ_G
     ↓
Earlier safe retirement
```

---

# TGS Algorithm

The implementation follows a simple temporal principle:

```text
Initialize with dense graph
        │
        ▼
Warmup / representation learning
        │
        ▼
Estimate edge retirement attractiveness
        │
        ▼
Protect dangerous / high-value edges
        │
        ▼
Retire low-score eligible edges
        │
        ▼
Repeat periodically
        │
        ▼
Stop at the learned / configured sparsity regime
```

A tuned configuration used in the verified benchmark was:

```yaml
warmup: 40
anneal_steps: 100
max_retire_frac: 0.02
max_sparsity: 0.50
retire_every: 5
```

Later experiments also evaluated matched final sparsities around 65%, depending on the benchmark protocol.

---

# Influence Estimator

The estimator was redesigned around measured relationships with exact edge influence `Ie(t)`.

At epoch 200, for 300 edges:

| Signal | Correlation with exact influence |
|---|---:|
| Degree product | **r = -0.648** |
| ER proxy | **r = +0.621** |
| Gradient EMA | ~0.000 |

The redesigned estimator therefore uses:

### Primary structural signal

```text
deg(u) × deg(v)
```

High degree-product edges are more attractive for retirement under the measured relationship.

### Bridge-edge gate

```text
er_proxy = 1/deg(u) + 1/deg(v)
```

The gate hard-locks the top 10% of bridge-like edges.

### Gradient modulation

Actively used edges receive protection from retirement.

### Maturity discount

Edges with stable representations / low gradient variance become more eligible for retirement.

Conceptually:

```text
score =
    (1 - structural_norm)
    × (1 + α × grad_norm)
    / (1 + γ × maturity_norm)
```

where:

```text
low score  = safer to retire
high score = more dangerous to retire
```

---

# Estimator Validation

The redesigned estimator substantially improved agreement with exact influence:

```text
Spearman r:
0.50–0.62
```

versus approximately:

```text
~0.00 before redesign
```

Bottom-20% agreement improved from:

```text
0.16–0.17
```

to:

```text
0.47–0.60
```

This is approximately a threefold improvement in both measures.

---

# Scheduler

The fixed-ε threshold was replaced by a percentile-based retirement gate.

Instead of asking:

```text
score < ε ?
```

TGS asks:

```text
Which eligible edges are in the lowest score percentile?
```

The scheduler retires the lowest-scoring eligible edges, up to the configured maximum retirement fraction.

This has two practical advantages:

- no fixed epsilon needs to be tuned for every graph;
- the policy adapts to the score distribution of each graph.

The implemented gate uses the bottom 30th percentile as the candidate pool.

---

# JSD Disagreement Signal

Degree-product structure is powerful, but it becomes weak on nearly uniform-degree graphs.

TGS therefore adds a model-state signal:

**Jensen–Shannon divergence between endpoint predictions.**

After an evaluation forward pass, each edge receives:

```text
JSD(p_u, p_v)
```

High JSD means the endpoints currently receive different class predictions and therefore may represent contradictory message-passing signals.

### Adaptive weighting

For graphs with:

```text
deg_cv < 0.2
```

disagreement receives high weight:

```text
0.85
```

For hub-structured graphs with:

```text
deg_cv >= 0.5
```

structural information remains dominant and disagreement is an additive signal:

```text
0.25
```

### Measured impact

Texas:

```text
t = -0.8
     ↓
t = -2.6, p < 0.05
```

Minesweeper moved from approximately random-level sparsification performance to consistently beating random sparsification.

---

# Benchmark Results

## Verified configurations

### Cora

```text
Test accuracy:     0.801
Sparsity:          26.1%
FLOPs reduction:   19.7%
Distortion bound:  0.028
```

### CiteSeer

```text
Test accuracy:     0.655
Sparsity:          11.4%
FLOPs reduction:   9.4%
Distortion bound:  0.010
```

---

## Cora comparison

At one evaluation point:

| Method | Test accuracy | Sparsity |
|---|---:|---:|
| Dense GCN | 0.810 | 0% |
| TGS | 0.801 | 26.1% |
| Random @10% | 0.805 | 10% |
| Effective Resistance @10% | **0.820** | 10% |

The effective-resistance result demonstrates an important caveat: TGS is not universally the highest-accuracy method at every sparsity level. Its central contribution is the temporal mechanism and the ability to reach strong accuracy under substantial sparsification.

---

# Matched-Sparsity Benchmark

Seed 42, 300 epochs:

| Dataset | Method | Accuracy | Sparsity | FLOPs ↓ |
|---|---|---:|---:|---:|
| Cora | **TGS** | **0.801** | 61.2% | **51.7%** |
| Cora | Dense GCN | 0.810 | 0% | 0% |
| Cora | Random | 0.709 | 61.2% | 61.2% |
| Cora | Local Degree | 0.745 | 61.2% | 61.2% |
| CiteSeer | **TGS** | **0.654** | 56.9% | **48.2%** |
| CiteSeer | Dense GCN | 0.717 | 0% | 0% |
| CiteSeer | Random | 0.661 | 56.9% | 56.9% |
| CiteSeer | Local Degree | 0.661 | 56.9% | 56.9% |
| PubMed | **TGS** | **0.756** | 65.1% | **54.9%** |
| PubMed | Dense GCN | 0.790 | 0% | 0% |
| PubMed | Random | 0.704 | 65.1% | 65.1% |
| PubMed | Local Degree | 0.761 | 65.1% | 65.1% |

TGS substantially outperforms random pruning at matched sparsity and remains close to dense performance on the benchmark datasets.

The reported dense gaps are:

- Cora: <1%
- CiteSeer: <7%
- PubMed: <3.4%

while using approximately **49–55% fewer FLOPs** than dense training in these matched-sparsity runs.

---

# TGS-Selected Edges vs Static Baselines

At matched sparsity of approximately 65%:

| Dataset | TGS fresh edges | Effective Resistance | Random |
|---|---:|---:|---:|
| Cora | **0.730** | 0.685 | 0.700 |
| CiteSeer | **0.664** | 0.651 | 0.661 |

A fresh GCN trained from scratch on the TGS-selected edge set outperforms the static baselines.

This is an important validation because it separates:

```text
"the original TGS training happened to work"
```

from:

```text
"the topology selected by TGS is itself useful."
```

---

# Real-World Graph Results

## Wisconsin

Graph:

```text
n = 251
m = 515
homophily = 0.196
deg_cv = 1.01
```

Five seeds:

| Method | Accuracy | Std |
|---|---:|---:|
| **TGS** | **0.655** | 0.011 |
| Dense | 0.533 | 0.016 |
| Random @65% | 0.569 | 0.015 |
| Local Degree @65% | 0.647 | 0.012 |
| Effective Resistance @65% | 0.588 | 0.018 |

TGS:

```text
5/5 seed wins
t = -10.6
p < 0.001
```

TGS improves over dense by approximately:

```text
+12.2pp
```

while removing approximately 65% of edges.

This is the largest real-dataset TGS margin in the reported experiments.

---

## Wisconsin sparsity sweep

Six sparsity levels were tested with five seeds per condition.

TGS:

- beats every static baseline at every non-zero sparsity level;
- exceeds dense accuracy from 20% sparsity onward;
- rises from **0.592 at 20% sparsity** to **0.718 at 80% sparsity**;
- increases its margin over the best static baseline from **+0.025 at 35%** to **+0.047 at 80%**.

This is especially notable because Wisconsin is strongly heterophilous.

---

## Chameleon

```text
n = 2,277
m = 36,101
homophily = 0.235
deg_cv = 2.93
```

Five seeds:

| Method | Accuracy | Std |
|---|---:|---:|
| **TGS** | **0.443** | 0.007 |
| Dense | 0.395 | 0.024 |
| Random @65% | 0.397 | 0.019 |
| Local Degree @65% | 0.441 | 0.015 |
| Effective Resistance @65% | 0.425 | 0.017 |

TGS:

```text
5/5 wins
t = -3.9
p < 0.01
```

---

## Texas

```text
n = 183
m = 325
homophily = 0.108
deg_cv = 1.07
```

Ten seeds:

| Method | Accuracy | Std |
|---|---:|---:|
| **TGS** | **0.692** | 0.014 |
| Dense | 0.659 | 0.034 |
| Random @65% | 0.568 | 0.021 |
| Local Degree @65% | 0.487 | 0.029 |

TGS:

```text
8/10 seed wins
t = -2.6
p < 0.05
```

Margins:

```text
TGS vs random:       +12.4pp
TGS vs local degree: +20.5pp
TGS vs dense:         +3.2pp
```

The JSD disagreement signal improved the dense comparison from `t=-0.8` to `t=-2.6`.

---

## Minesweeper

Because the classes are approximately 80/20, ROC-AUC is used rather than raw accuracy.

```text
n = 10,000
m = 78,804
homophily = 0.683
deg_cv = 0.075
```

| Method | Mean AUC |
|---|---:|
| **TGS** | **0.710** |
| Dense | 0.712 |
| Random @65% | 0.697 |
| Local Degree @65% | 0.593 |
| Effective Resistance @65% | 0.697 |

TGS matches dense performance within approximately `0.002 AUC` while using 65% fewer edges.

It beats random sparsification on all five seeds.

Minesweeper is outside the main structural winning zone because its degree distribution is nearly uniform. This motivates the model-state JSD disagreement component.

---

# Two-Regime Rule

Across the six initially evaluated real datasets, a refined rule was also tested:

```text
Homophilic regime (h > 0.5):
    TGS wins iff h × CV > 1.0

Heterophilic regime (h <= 0.5):
    TGS wins iff CV > 0.8
```

The six-dataset validation reported:

```text
Accuracy = 6/6 = 100%
```

The datasets included:

| Dataset | h | CV | Score | TGS wins |
|---|---:|---:|---:|---|
| Cora | 0.810 | 1.341 | 1.0862 | Yes |
| CiteSeer | 0.736 | 1.236 | 0.9097 | No |
| PubMed | 0.802 | 1.653 | 1.3257 | Yes |
| Texas | 0.108 | 1.072 | 0.1158 | Yes |
| Wisconsin | 0.196 | 1.009 | 0.1978 | Yes |
| Chameleon | 0.234 | 1.800 | 0.4212 | Yes |

This rule should be treated as a reported empirical refinement, not as a replacement for the separately frozen prospective threshold.

---

# Hub × Cross-Class Interaction

The 2×2 intervention on Wisconsin directly manipulates cross-class density and hub concentration.

The four cells were:

| Cell | Regime | TGS | Dense | Dense − TGS | TGS wins |
|---|---|---:|---:|---:|---:|
| A | Low cross × Low hub | 0.655 | 0.659 | +0.004 | 1/5 |
| B | Low cross × High hub | 0.655 | 0.659 | +0.004 | 1/5 |
| C | High cross × Low hub | 0.655 | 0.651 | -0.004 | 2/5 |
| D | **High cross × High hub** | **0.655** | **0.584** | **-0.071** | **5/5** |

Regression effects:

```text
β_cross = -0.041
β_hub   = -0.033
β_int   = -0.067
```

The high-cross/high-hub regime reproduces the Wisconsin-like mechanism.

The interaction is therefore not merely descriptive:

```text
Cross-class noise × hub concentration
              ↓
        larger TGS advantage
```

---

# Architecture Generalization

TGS was evaluated across GCN, GAT, and GraphSAGE.

| Dataset | Architecture | Dense | TGS | Fresh | Sparsity | FLOPs ↓ |
|---|---|---:|---:|---:|---:|---:|
| Cora | GCN | 0.810 | 0.801 | 0.730 | 65.1% | 54.8% |
| Cora | GAT | 0.828 | 0.815 | **0.759** | 56.9% | 48.2% |
| Cora | SAGE | 0.785 | 0.785 | 0.711 | 56.9% | 48.2% |
| CiteSeer | GCN | 0.717 | 0.654 | 0.664 | 65.1% | 54.8% |
| CiteSeer | GAT | 0.703 | 0.667 | — | 61.2% | 51.7% |

The reported results support architecture-agnostic behavior across the tested GCN, GAT, and GraphSAGE configurations.

---

# Ablations

## Schedule

Cosine sparsity scheduling outperformed the tested linear/step alternatives on sparsity:

```text
0.612 vs 0.569
```

with the same accuracy in the reported comparison.

---

## Warmup

Warmup is critical.

```text
warmup = 20 → accuracy 0.768
warmup >= 40 → accuracy 0.801
```

The independent timing sweep agrees:

```text
warmup=0   → 0.751
warmup=20  → 0.795
warmup=40  → 0.801
warmup>=60 → 0.801 plateau
```

Thus 40 epochs is not merely a tuned value; the timing experiments provide a mechanistic explanation for it.

---

## Retirement frequency

Retiring every epoch can reach approximately 65% sparsity while preserving 0.801 accuracy, with diminishing returns beyond that schedule.

---

## Sparsity ceiling

The method self-limits around approximately 61% in one benchmark.

Changing the ceiling from:

```text
0.65 → 0.75
```

did not improve the resulting sparsity/accuracy outcome in that experiment.

This suggests that the estimator/scheduler can naturally become conservative before the nominal maximum is reached.

---

# Component Ablation

Ground-truth metric: fresh GCN accuracy on the selected edge set.

| Variant | Ie(t) r | Bottom-20% agreement | Fresh accuracy | vs Random |
|---|---:|---:|---:|---:|
| Random | -0.024 | 0.178 | 0.691 | baseline |
| Degree only | +0.124 | 0.289 | 0.720 | +2.9pp |
| Degree + gradient | +0.052 | 0.289 | **0.730** | **+3.9pp** |
| Degree + maturity | +0.008 | **0.322** | 0.723 | +3.2pp |
| **Full TGS** | **+0.052** | 0.289 | **0.730** | **+3.9pp** |

Measured roles:

- **Degree:** primary structural driver.
- **Gradient:** protects actively used edges.
- **Maturity:** identifies stable/converged redundancy.
- **Full estimator:** best fresh accuracy in this ablation.

The results support retaining each component for a measured reason rather than as an arbitrary engineering choice.

---

# Temporal vs Static Proof

At matched 65% sparsity on Cora:

| Method | Accuracy | FLOPs | Runtime |
|---|---:|---:|---:|
| Dense GCN | 0.810 | 100% | 100% |
| Static degree prune | 0.726 | 35% | 89% |
| **TGS** | **0.801** | **45%** | 110% |

TGS vs static:

```text
+7.5pp accuracy
+21% runtime
```

The central tradeoff is explicit:

> TGS spends additional training-time computation to preserve the representation-learning window, then finishes with a substantially smaller graph.

---

# Runtime and Memory

Real Cora measurements:

| Method | Total time | Memory | Inference |
|---|---:|---:|---:|
| Dense | 9.4s | 100% | 8.21ms |
| **TGS** | 10.3s | **13%** | **7.22ms** |
| Static | 8.3s | 8% | 7.53ms |

TGS therefore:

- reduces memory footprint by approximately **87% vs dense**;
- produces **faster inference** than dense in the reported measurement;
- incurs approximately **13% per-epoch overhead** during training.

Overhead:

```text
Influence estimator: 3.46ms / epoch  (~10.7%)
Scheduler + guard:   1.39ms / epoch  (~4.3%)
```

The reported total training-time overhead is approximately 13% per epoch.

The key systems property is that the active edge tensor shrinks throughout training.

---

# Practical Decision Rule

For prospective deployment, the strongest validated frozen rule is:

```text
SCORE(G) = homophily(G) × deg_cv(G)

if SCORE(G) > 0.0505:
    TGS is predicted to provide a substantive advantage
else:
    TGS is predicted to provide little or no advantage
```

This threshold was locked after 20-graph calibration.

Do not silently retune this threshold on the evaluation set if the goal is prospective validation.

The later two-regime rule is an additional empirical refinement and should be reported separately rather than conflated with the frozen prospective threshold.

# Conclusion

TGS is built around a deliberately simple observation:

> **An edge can be important before representation learning and redundant after representation learning.**

Static pruning must decide before the model knows.

TGS waits.

The experimental evidence supports a complete chain:

```text
GRAPH STRUCTURE
      │
      │  H × CV
      ▼
MATURATION SPEED
      │
      │  probe@20: r = +0.944
      ▼
TASK-RELEVANT REPRESENTATIONS
      │
      │  edges become redundant
      ▼
SAFE RETIREMENT WINDOW
      │
      │  τ_G: Spearman ρ = -0.728
      ▼
TEMPORAL SPARSIFICATION
      │
      │  +16pp from t=0 → t=20
      ▼
SPARSE GRAPH
      │
      ├── lower FLOPs
      ├── dramatically lower memory
      └── preserved or improved accuracy
```

The most important result is not that one estimator beats another.

It is the intervention:

```text
Same topology.
Same final sparsity.
Same graph.
Different retirement time.

t = 0     → 0.750
t >= 20   → 0.910

Δ = +16pp
```

And the temporal-order ablation strengthens the conclusion:

```text
TGS order       0.818
Random order    0.819
Reverse order   0.814
Static          0.730
```

The individual ranking of edges is therefore not the whole story.

**The temporal window is.**

TGS turns sparsification from a one-shot topology-selection problem into a **training-time decision about when information has become safely compressible**.

That is the central contribution.
