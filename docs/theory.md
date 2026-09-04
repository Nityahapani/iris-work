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
| Proposition 9.1 | FLOPs savings: `1 - (1/T)Σmt/m0` |\

## Implementation Correspondence

| Theory | Implementation |
|---|---|
| `TemporalGraph` sequence `{G_t}` | `tgs/core/temporal_graph.py` |
| Retirement schedule τ(e) | `TemporalGraph._retirement_step` |
| Influence `Ie(t)` | `tgs/core/influence.py` — `GradientNormEstimator` |
| Safe retirement criterion (Def 5.1) | `tgs/schedulers/retirement_scheduler.py` |
| Adaptive ε (Section 8 connection) | `tgs/schedulers/adaptive_scheduler.py` |
| FLOPs bound (Prop 9.1) | `tgs/evaluation/flops.py` |

---

## Theory–Implementation Gap: Threshold vs Rank-Based Retirement

**Identified:** empirically confirmed via `experiments/edge_lifetime.py` and score-at-retirement
measurement. **Status:** known limitation, documented here for honesty. Does not affect the
experimental results but does affect which theoretical claims can be made.

### The gap

Theorems 5.2, 5.3, and 6.3 are stated for the *absolute-threshold* retirement rule:

> **Definition 5.1 (safe retirement criterion):** retire edge e at step t if Iₑ(t) ≤ ε.

Under this rule, every retired edge satisfies Iₑ(t) ≤ ε by construction, so the cumulative
distortion bound from Theorem 6.3:

> ‖H_t − H_t^{−S}‖_F ≤ k · ε

follows directly (k retired edges, each contributing at most ε).

The actual scheduler (`RetirementScheduler.step`) implements a **rank-based** rule instead:

> retire the bottom 30% of active influence scores, subject to rate and sparsity limits.

This is a relative criterion: edges are retired if they rank in the bottom 30% of the
*current active set*, regardless of their absolute score. Empirical measurement confirms
the gap is not small:

| Metric | Value |
|---|---|
| Retired edges (Wisconsin, seed 42) | 350 / 515 |
| Edges satisfying Iₑ(t) ≤ ε at retirement | 107 / 350 (30.6%) |
| Edges with score > ε at retirement | 243 / 350 (69.4%) |
| Mean score / ε ratio at retirement | 54.1× |
| Max score / ε ratio at retirement | 171× |

So the implementation retires edges whose scores are up to 171× above the theoretical
threshold ε. The distortion bound ‖H_t − H_t^{−S}‖_F ≤ k · ε does **not** hold as stated
when retired edges have Iₑ(t) >> ε.

### What does hold

**Theorem 4.4 and Theorem 5.2** remain valid as theoretical results — they characterise
the distortion incurred by retiring a *specific* edge at the moment its influence crosses ε.
These theorems are not vacuous; they motivate the scoring function and establish that
*if* the scheduler retires only sub-threshold edges, the distortion is controlled.

**The rank-based rule has a weaker but still meaningful guarantee.** Let Iₑ^(q)(t) denote
the q-th percentile of active influence scores at step t. The scheduler retires edges
satisfying Iₑ(t) ≤ Iₑ^(0.30)(t). The actual distortion incurred is then:

> ‖H_t − H_t^{−S}‖_F ≤ Σ_{e ∈ R_t} Iₑ(t)

(sum over edges retired at step t, from Theorem 4.4 applied additively). This bound
holds exactly regardless of the retirement rule, because it uses actual scores, not ε.
The implementation tracks `cumulative_distortion_bound = k · ε`, which is an invalid
proxy — it should track `Σ Iₑ(t)` instead.

**Proposition 9.1 (FLOPs savings)** is unaffected — it depends only on mt (edges active
at step t), not on the retirement criterion.

**The experimental results** (accuracy, timing, CV→τ* sweep, edge lifetime) are all
unaffected — they are empirical measurements, not claims derived from the distortion bound.

### The correct claim to make

Replace:

> "TGS guarantees that every retired edge has influence ≤ ε,
>  so cumulative distortion is bounded by k · ε."

With:

> "TGS ranks active edges by estimated influence and retires the lowest-scoring edges
>  subject to a sparsification budget. The cumulative representation distortion is bounded
>  by Σ_{e ∈ S} Iₑ(τ(e)), where the sum runs over all retired edges and τ(e) is the
>  retirement epoch (Theorem 4.4, applied additively). The rank-based rule makes this
>  bound data-adaptive: it tightens automatically as training progresses and the
>  lowest-scoring edges approach zero influence."

### Fix options (ranked by effort)

**Option A — Re-state Theorem 6.3 for the rank rule (recommended).**
The additive distortion bound Σ Iₑ(τ(e)) is correct for any retirement rule.
Re-derive the theorem in terms of this sum rather than k · ε. No code changes needed.

**Option B — Fix the implementation to match the threshold rule.**
Change `RetirementScheduler.step` to retire exactly the edges satisfying Iₑ(t) ≤ ε,
not a fixed percentile. The adaptive ε schedule already exists to modulate this.
Downside: behaviour may change significantly if ε is poorly calibrated (could retire
nothing for many steps, then burst-retire many at once).

**Option C — Hybrid: rank gate + absolute cap.**
Retire edges in the bottom q% AND Iₑ(t) ≤ ε_max. This ensures no edge with very high
influence is ever retired (preserving a weaker form of the distortion guarantee) while
keeping the smoothness of rank-based retirement. Requires tuning both q and ε_max.

**Option D — Track actual distortion, not k · ε.**
Minimal code fix: replace `cumulative_distortion_bound = k * self.epsilon` with
`cumulative_distortion_bound += scores_at_retirement.sum()`. This makes the reported
bound valid (from Theorem 4.4) without changing the retirement rule or any theorems.

The recommended path for ISEF presentation: implement Option D immediately (one-line fix,
makes the reported metric honest), and restate Theorem 6.3 as Option A describes.
The rank-based rule is defensible as a practical algorithm; the gap is in how the
distortion of that rule was characterised, not in the algorithm itself.
