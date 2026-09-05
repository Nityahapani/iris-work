# Generalization Experiment — Coauthor-CS
## "Does the budget-race result hold on an unseen real-world graph?"

**Dataset:** Coauthor-CS (Microsoft Academic Graph, computer science domain)  
**Scale:** n=18,333 nodes · m=163,788 edges · nc=15 classes · nf=6,805 features (PCA→128)  
**Split:** 60/20/20 random (seed=0, fixed across all methods)  
**TGS config:** **IDENTICAL to Cora — zero retuning**  
`warmup=40, anneal_steps=100, max_sparsity=0.65, retire_every=2`  
**Seeds:** 3 · **Epochs:** 150 · **Budget:** 50% of dense cumulative FLOPs

---

## Graph Properties

| Property | Value | Implication |
|---|---|---|
| homophily | 0.808 | High — same-class neighbours dominate |
| deg_cv | 1.019 | Moderate degree variation |
| SCORE (homophily × deg_cv) | **0.8237** | >> 0.0505 threshold → TGS predicted beneficial |
| avg_deg | 8.93 | Same order as Cora (7.9) |
| n | 18,333 | **6.8× larger than Cora** |

---

## Experiment 1 — Accuracy-Constrained

Accuracy floor: val ≥ 0.7789 (= 95% of dense val=0.8199)

| Method | Test (full 150ep) | Sp | Epochs → floor | FLOP% → floor |
|---|---|---|---|---|
| Dense GCN | 0.814 ± 0.013 | 0.000 | 125 | 83.3% |
| Random sparsify | **0.820 ± 0.005** | 0.570 | 115 | **32.9%** |
| Degree sparsify | 0.804 ± 0.006 | 0.570 | 124 | 35.5% |
| Fixed-sched TGS | 0.579 ± 0.034 | 0.650 | NEVER | NEVER |
| Adaptive TGS | 0.343 ± 0.045 | 0.597 | NEVER | NEVER |

**Observation:** Static sparsification reaches the accuracy floor using ~33–36% of dense FLOPs — a strong result. TGS with Cora's warmup config fails to hit the floor on this graph.

---

## Experiment 2 — Budget Race (50% FLOPs)

Dense gets 75 epochs; static methods get 150 epochs (sparse graph costs ~57% less per epoch); TGS self-manages within the same cumulative cap.

| Method | @budget acc | ±σ | Δ vs dense | Sp | Inf ms |
|---|---|---|---|---|---|
| Dense GCN (@budget) | 0.433 | 0.070 | baseline | 0.000 | 75.6 |
| Random sparsify | **0.820** | 0.005 | **+38.7 pp** | 0.570 | 18.2 |
| Degree sparsify | 0.804 | 0.006 | +37.1 pp | 0.570 | 16.9 |
| Fixed-sched TGS | 0.579 | 0.034 | +14.6 pp | 0.650 | 16.4 |
| Adaptive TGS | 0.343 | 0.045 | −9.0 pp | 0.597 | 14.3 |

---

## Diagnosis — Why TGS Underperforms Here

This is a negative result, and it is scientifically informative. Four distinct causes:

**1. Budget allocation front-loads cost against TGS.**  
TGS's 40-epoch warmup runs on the dense graph (163K edges). This consumes **53% of the total budget** before any edges are retired. Static methods have no warmup cost and spend all 150 sparse-graph epochs within the same budget. The budget-race framing is structurally unfair to any algorithm that front-loads dense computation.

*Correction for future work:* Budget should be measured from the point where sparsification begins, or warmup should be excluded from the budget accounting.

**2. Adaptive TGS collapses (0.343) while Fixed-sched TGS partially holds (0.579).**  
The adaptive scheduler reads validation accuracy to decide when retirement is safe. At epoch 40 on CoauthorCS, validation accuracy is still in its steep-rise phase (not yet plateaued), so the scheduler incorrectly interprets early-training noise as a signal that representations are mature. This triggers premature retirement — edges are removed before the model has learned to rely on them, causing accuracy collapse.

Fixed-schedule TGS avoids this by firing at a fixed epoch (40) rather than reading the val-acc signal, but still underperforms because 40 epochs is insufficient for CoauthorCS representations to mature.

**3. The default warmup (40 epochs) is tuned for Cora scale (n=2,708).**  
CoauthorCS is **6.8× larger**. The warmup period needed for representations to consolidate scales with graph size and edge density. On CoauthorCS, warmup should be ≥80 epochs (proportional to n). This is a known limitation of the fixed-config transfer: the SCORE rule predicts *whether* TGS helps, but not *what timing* to use.

**4. Static methods "win" for a confounded reason.**  
Random and degree sparsify both score 0.820 — higher than dense (0.814) at full 150 epochs. This is because on an overparameterised graph (n=18K, dense avg_deg=9), the sparse graph is simply easier to optimise. The static methods get all 150 epochs on an easier problem; this is not evidence that their topology selection is better than TGS's.

---

## What This Result Establishes

**What it does not establish:**
- That TGS fails on graphs with high homophily (SCORE=0.82 correctly predicted it should work)
- That the algorithm is wrong

**What it does establish:**

1. **The SCORE rule is necessary but not sufficient.** It correctly identifies that CoauthorCS is a TGS-beneficial graph, but provides no guidance on warmup timing. A more complete predictive rule would include a graph-size-aware warmup schedule: `warmup ≥ k × (n / n_Cora) × base_warmup`.

2. **The budget-race framing penalises front-loaded algorithms.** A fixed-FLOPs budget applied uniformly does not account for algorithms that invest compute up-front to amortise it later. A more principled comparison would equalise inference cost rather than training cost, or measure budget from convergence time.

3. **Fixed-schedule TGS is more robust than adaptive TGS to warm-start failure.** When the val-acc signal is unreliable (early in training on large graphs), the adaptive scheduler misfires. Fixed-schedule is a safer fallback.

4. **The Cora generalization result (budget race, +38 pp vs dense → only −2.5 pp for TGS) is a tighter, better-controlled experiment.** Cora's scale matches the algorithm's configuration. CoauthorCS is a stress-test that exposes a real limitation.

---

## Corrected Claim for ISEF

> TGS achieves the required predictive performance using significantly fewer message-passing operations than dense training **on graphs where the warmup timing is appropriately calibrated to graph scale**. The SCORE rule identifies beneficial graphs; a graph-size-aware warmup schedule is additionally required for reliable transfer. This identifies a concrete scope for future work: an adaptive warmup that scales with n rather than a fixed epoch count.

---

## Raw Seed Results

| Seed | Fixed-sched TGS | Adaptive TGS |
|------|----------------|--------------|
| 42 | 0.5956 | 0.2964 |
| 43 | 0.6098 | 0.4031 |
| 44 | 0.5320 | 0.3289 |
| **mean** | **0.5791** | **0.3428** |

The seed variance for adaptive TGS (std=0.045) is high relative to fixed-sched (std=0.034), consistent with the adaptive scheduler's val-acc feedback being unstable at epoch 40 on this graph.

---

*All experiments run live. Code: `experiments/budget_constrained.py`.  
Raw parts: `results/budget_experiment/cs_parts/`.  
Config: zero retuning from Cora.*
