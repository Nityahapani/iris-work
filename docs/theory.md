# Theoretical Framework

The full theoretical framework (proofs, definitions, and theorems) is maintained separately as a LaTeX document.

## Summary of Key Results

| Result | Statement |
|---|---|
| Theorem 4.4 | Single-edge removal: `Ie(t) ≤ CH‖Δe‖F` |
| Theorem 5.2 | Retirement at `Ie(t) ≤ ε` guarantees `‖H_τ - H_τ^{-e}‖F ≤ ε` |
| Theorem 5.3 | Prediction distortion: `‖Ŷ_τ - Ŷ_τ^{-e}‖F ≤ Kf·ε` |
| Theorem 6.3 | k retirements: cumulative distortion `≤ kε` |
| Proposition 7.2 | Temporal sparsification strictly generalises all static methods |
| Theorem 7.4 | Optimal retirement: `Be(τ*) = λ` |
| Proposition 8.2 | Redundant edges retire in finite steps under convergence |
| Proposition 9.1 | FLOPs savings: `1 - (1/T)Σmt/m0` |

## Implementation Correspondence

| Theory | Implementation |
|---|---|
| `TemporalGraph` sequence `{G_t}` | `tgs/core/temporal_graph.py` |
| Retirement schedule τ(e) | `TemporalGraph._retirement_step` |
| Influence `Ie(t)` | `tgs/core/influence.py` — `GradientNormEstimator` |
| Safe retirement criterion (Def 5.1) | `tgs/schedulers/retirement_scheduler.py` |
| Adaptive ε (Section 8 connection) | `tgs/schedulers/adaptive_scheduler.py` |
| FLOPs bound (Prop 9.1) | `tgs/evaluation/flops.py` |
