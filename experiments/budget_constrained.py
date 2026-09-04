"""
experiments/budget_constrained.py
==================================
Adaptive Compute-Budget Experiment for TGS — ISEF Software Design

Research question
-----------------
"Can TGS make graph neural networks usable when compute or latency is
actually constrained?"

Three complementary experiments
--------------------------------
1. ACCURACY-CONSTRAINED   — Given a fixed accuracy floor (≥95% of dense
                            val performance), how much computation can each
                            method eliminate?

2. BUDGET-CONSTRAINED RACE — Every method gets exactly the same cumulative
                             message-passing FLOP budget (50% of dense).
                             Who achieves the highest accuracy?

3. ACCURACY-PER-FLOP FRONTIER — Pareto curve of accuracy vs FLOPs used,
                                sweeping the sparsity / timing hyperparameter.
                                Headline metric: acc / (FLOPs / dense_FLOPs).

Baselines (all at same sparsity as final TGS graph)
------------------------------------------------------
  Dense GCN                — no sparsification
  Random sparsification    — remove edges uniformly at random, pre-training
  Degree pruning           — remove highest-degree-product edges, pre-training
  Effective-resistance     — remove lowest-ER edges (approx via random proj)
  Fixed-schedule TGS       — retire at fixed epoch regardless of training state
  Adaptive TGS             — full system (timing + structure-aware)

Datasets: Cora, CiteSeer, Texas, Wisconsin (all 4 from main experiments)
Seeds: 5 seeds per method for significance
Epochs: 300

Results saved to results/budget_experiment/
"""

import sys, os, time, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.transforms import NormalizeFeatures

from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils.reproducibility import set_seed

DEVICE  = torch.device("cpu")
EPOCHS  = 300
SEEDS   = [42, 43, 44, 45, 46]
N_LAYERS = 2
HIDDEN   = 64
BUDGET_FRAC = 0.50        # fraction of dense FLOPs given as budget
ACC_FLOOR   = 0.95        # fraction of dense val acc required

OUT_DIR = "results/budget_experiment"
os.makedirs(OUT_DIR, exist_ok=True)

# TGS default config (from main experiments)
TGS_CFG = dict(
    epsilon_max=5e-3, epsilon_min=1e-5, anneal_steps=100,
    warmup_steps=40, max_retire_frac=0.10,
    max_sparsity=0.65, retire_every=2
)

# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def load_dataset(name):
    root = "./data"
    if name in ("Cora", "CiteSeer", "PubMed"):
        ds = Planetoid(root=root, name=name, transform=NormalizeFeatures())
        return ds[0], ds.num_features, ds.num_classes
    if name in ("Texas", "Wisconsin", "Cornell"):
        ds = WebKB(root=root, name=name, transform=NormalizeFeatures())
        d = ds[0]
        d.train_mask = d.train_mask[:, 0]
        d.val_mask   = d.val_mask[:, 0]
        d.test_mask  = d.test_mask[:, 0]
        return d, ds.num_features, ds.num_classes
    raise ValueError(name)

# ─────────────────────────────────────────────────────────────────
# Effective-resistance sparsification (random projection approx)
# ─────────────────────────────────────────────────────────────────

def effective_resistance_scores(edge_index, num_nodes, k=64, seed=42):
    """
    Approximate effective resistance for each edge via random projection.
    ER(u,v) = (L†_uu + L†_vv - 2L†_uv) ≈ ||L†^{1/2}(e_u - e_v)||²
    We approximate L†^{1/2} R ≈ solve((L + εI), R) for random R.
    Returns: array of shape [m] with ER score per edge.
    """
    rng = np.random.RandomState(seed)
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    m   = len(row)
    vals = np.ones(m)
    A = sp.csr_matrix((vals, (row, col)), shape=(num_nodes, num_nodes))
    deg = np.array(A.sum(1)).ravel()
    L   = sp.diags(deg) - A
    R   = rng.randn(num_nodes, k) / np.sqrt(k)
    Lp  = spla.spsolve(L + sp.eye(num_nodes) * 1e-6, R)   # [n, k]
    er  = np.sum((Lp[row] - Lp[col]) ** 2, axis=1)        # [m]
    return er

# ─────────────────────────────────────────────────────────────────
# Core training loop — returns rich result dict
# ─────────────────────────────────────────────────────────────────

def _flops(mt): return N_LAYERS * mt * HIDDEN * 2

def train_dense(data, nf, nc, seed, epochs=EPOCHS):
    """Dense GCN — the reference."""
    set_seed(seed)
    m0 = data.edge_index.shape[1]
    model = TemporalGCN(nf, HIDDEN, nc, N_LAYERS, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    dense_flops_per_ep = _flops(m0)

    epoch_times, val_accs, cum_flops = [], [], []
    best_val = best_test = 0.0
    cum = 0

    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        logits = model(data.x, data.edge_index)
        loss   = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(data.x, data.edge_index).argmax(-1)
        val  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        cum += dense_flops_per_ep
        val_accs.append(val); cum_flops.append(cum)
        epoch_times.append(time.perf_counter() - t0)
        if val > best_val: best_val, best_test = val, test

    inf_times = []
    model.eval()
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter(); model(data.x, data.edge_index)
            inf_times.append(time.perf_counter() - t0)

    return dict(
        method="Dense GCN", best_val=best_val, best_test=best_test,
        sparsity=0.0, total_flops=cum, dense_flops=cum,
        flops_used_pct=100.0, acc_per_flop=best_test / (cum / 1e9),
        total_time_s=sum(epoch_times),
        inf_ms=np.median(inf_times) * 1000,
        val_curve=val_accs, flops_curve=cum_flops,
        m0=m0, final_m=m0, epochs_to_target_acc=None,
    )


def train_static(data, nf, nc, seed, target_sparsity, method_name,
                 score_fn, epochs=EPOCHS):
    """
    Static sparsification: remove edges before training.
    score_fn(edge_index, num_nodes) → scores array, higher = more important.
    Edges with LOWEST scores are removed.
    """
    set_seed(seed)
    m0  = data.edge_index.shape[1]
    scores = score_fn(data.edge_index, data.num_nodes)
    n_keep = int(m0 * (1 - target_sparsity))
    keep_idx = np.argsort(-scores)[:n_keep]          # keep highest-score edges
    keep_mask = torch.zeros(m0, dtype=torch.bool)
    keep_mask[keep_idx] = True
    ei = data.edge_index[:, keep_mask]
    final_m = ei.shape[1]

    model = TemporalGCN(nf, HIDDEN, nc, N_LAYERS, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    ep_flops = _flops(final_m)
    dense_total = _flops(m0) * epochs

    epoch_times, val_accs, cum_flops = [], [], []
    best_val = best_test = 0.0
    cum = 0
    epochs_to_target = None

    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        logits = model(data.x, ei)
        loss   = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(data.x, ei).argmax(-1)
        val  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        cum += ep_flops
        val_accs.append(val); cum_flops.append(cum)
        epoch_times.append(time.perf_counter() - t0)
        if val > best_val: best_val, best_test = val, test

    inf_times = []
    model.eval()
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter(); model(data.x, ei)
            inf_times.append(time.perf_counter() - t0)

    return dict(
        method=method_name, best_val=best_val, best_test=best_test,
        sparsity=1 - final_m/m0, total_flops=cum, dense_flops=dense_total,
        flops_used_pct=cum/dense_total*100,
        acc_per_flop=best_test / (cum / 1e9),
        total_time_s=sum(epoch_times),
        inf_ms=np.median(inf_times) * 1000,
        val_curve=val_accs, flops_curve=cum_flops,
        m0=m0, final_m=final_m,
        epochs_to_target_acc=None,
    )


def train_tgs(data, nf, nc, seed, epochs=EPOCHS, fixed_retire_epoch=None,
              budget_cap_flops=None):
    """
    TGS training.
    fixed_retire_epoch: if set, retire all scheduled edges at that exact epoch
                        regardless of influence scores (Fixed-schedule TGS).
    budget_cap_flops: if set, stop updating (but keep evaluating) once cumulative
                      FLOPs exceed this budget (Budget-constrained race).
    """
    set_seed(seed)
    m0  = data.edge_index.shape[1]
    tg  = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE,
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    model = TemporalGCN(nf, HIDDEN, nc, N_LAYERS, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(
                list(model.parameters()) + [est.edge_weights],
                lr=0.01, weight_decay=5e-4)
    sched = AdaptiveRetirementScheduler(tg, **TGS_CFG)
    dense_total = _flops(m0) * epochs

    epoch_times, val_accs, cum_flops = [], [], []
    edge_counts = []
    best_val = best_test = 0.0
    cum = 0
    budget_exhausted = False
    epochs_to_target  = None    # epoch when model first hits dense_val * ACC_FLOOR

    # For fixed-schedule: compute how many edges to retire based on adaptive TGS logic
    # but fire all retirements at fixed_retire_epoch
    fixed_retire_done = False

    for ep in range(epochs):
        t0 = time.perf_counter()
        am = tg.active_mask

        # If budget exhausted, stop gradient updates but keep eval
        if budget_cap_flops and cum >= budget_cap_flops:
            if not budget_exhausted:
                budget_exhausted = True
            model.eval()
            with torch.no_grad():
                preds = model(data.x, tg.edge_index).argmax(-1)
            val  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
            test = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            val_accs.append(val); cum_flops.append(cum)
            edge_counts.append(tg.mt)
            epoch_times.append(time.perf_counter() - t0)
            if val > best_val: best_val, best_test = val, test
            continue

        # Fixed-schedule TGS: fire retirements all at once at fixed_retire_epoch
        if fixed_retire_epoch is not None and ep == fixed_retire_epoch and not fixed_retire_done:
            scores = est.influence_scores(am)
            # Retire until we reach target sparsity
            n_to_retire = int(m0 * TGS_CFG["max_sparsity"]) - (m0 - tg.mt)
            if n_to_retire > 0:
                eligible = am & (scores < float("inf"))
                eligible_idx = eligible.nonzero(as_tuple=False).squeeze(1)
                if len(eligible_idx) > 0:
                    s = scores[eligible_idx]
                    _, order = s.sort()
                    retire_idx = eligible_idx[order[:n_to_retire]]
                    tg.retire_edges(retire_idx)
            fixed_retire_done = True

        model.train()
        logits = model(data.x, tg.edge_index, est.edge_weights[am])
        loss   = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad(); loss.backward()
        est.update_influence(am)
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(data.x, tg.edge_index).argmax(-1)
        val  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        # Disagreement update
        with torch.no_grad():
            full_logits = model(data.x, data.edge_index)
            est.update_disagreement(full_logits, am)

        sched.update_val_acc(val)
        if fixed_retire_epoch is None:   # only auto-retire for adaptive TGS
            sched.step(est.influence_scores(am))

        cum += _flops(tg.mt)
        tg.step()

        val_accs.append(val); cum_flops.append(cum)
        edge_counts.append(tg.mt)
        epoch_times.append(time.perf_counter() - t0)
        if val > best_val: best_val, best_test = val, test

    inf_times = []
    model.eval(); final_ei = tg.edge_index
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter(); model(data.x, final_ei)
            inf_times.append(time.perf_counter() - t0)

    return dict(
        method="Fixed-schedule TGS" if fixed_retire_epoch else "Adaptive TGS",
        best_val=best_val, best_test=best_test,
        sparsity=tg.sparsity, total_flops=cum, dense_flops=dense_total,
        flops_used_pct=cum/dense_total*100,
        acc_per_flop=best_test / max(cum / 1e9, 1e-12),
        total_time_s=sum(epoch_times),
        inf_ms=np.median(inf_times) * 1000,
        val_curve=val_accs, flops_curve=cum_flops,
        edge_trajectory=edge_counts,
        m0=m0, final_m=tg.mt,
        budget_exhausted_at=None,
        epochs_to_target_acc=epochs_to_target,
    )

# ─────────────────────────────────────────────────────────────────
# Score functions for static baselines
# ─────────────────────────────────────────────────────────────────

def score_random(edge_index, num_nodes):
    m = edge_index.shape[1]
    return np.random.rand(m)    # random — caller uses fixed seed via set_seed

def score_degree(edge_index, num_nodes):
    """Low degree-product = low score = removed first."""
    row = edge_index[0].numpy(); col = edge_index[1].numpy()
    deg = np.bincount(np.concatenate([row, col]), minlength=num_nodes).astype(float)
    return deg[row] * deg[col]   # keep high-product (hub) edges

def score_er(edge_index, num_nodes):
    """High ER = important bridge = high score = keep."""
    return effective_resistance_scores(edge_index, num_nodes, k=64)

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 1 — Accuracy-constrained: how much compute to hit floor?
# ─────────────────────────────────────────────────────────────────

def exp_accuracy_constrained(datasets):
    print("\n" + "="*72)
    print("EXPERIMENT 1 — ACCURACY-CONSTRAINED COMPUTE REDUCTION")
    print(f"  Target: maintain ≥{ACC_FLOOR*100:.0f}% of dense val accuracy")
    print("="*72)

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)
        m0 = data.edge_index.shape[1]
        dense_total_flops = _flops(m0) * EPOCHS

        # ── Step 1: Run Dense to get the reference accuracy & val curve
        print("  [Dense]", end=" ", flush=True)
        dense_seeds = []
        for seed in SEEDS:
            r = train_dense(data, nf, nc, seed)
            dense_seeds.append(r)
            print(".", end="", flush=True)
        print()
        dense_val_mean = np.mean([r["best_val"] for r in dense_seeds])
        dense_test_mean = np.mean([r["best_test"] for r in dense_seeds])
        acc_floor_val = dense_val_mean * ACC_FLOOR
        print(f"  Dense val={dense_val_mean:.3f} test={dense_test_mean:.3f}")
        print(f"  Accuracy floor (≥{ACC_FLOOR*100:.0f}%): val≥{acc_floor_val:.3f}")

        # For each method: find the epoch at which it first clears the floor,
        # and report the cumulative FLOPs spent at that point.
        methods = {
            "Random sparsify":       (train_static, score_random),
            "Degree sparsify":       (train_static, score_degree),
            "Eff. resistance":       (train_static, score_er),
            "Fixed-schedule TGS":    (train_tgs, 40),     # retire at epoch 40
            "Adaptive TGS":          (train_tgs, None),
        }

        ds_results = {"dense": {
            "val_mean": dense_val_mean, "test_mean": dense_test_mean,
            "flops": dense_total_flops, "flops_pct": 100.0,
        }}

        # Determine TGS final sparsity first (from adaptive run) to match all baselines
        print(f"  [Adaptive TGS] calibration run...", end=" ", flush=True)
        cal = train_tgs(data, nf, nc, seed=42)
        tgs_sparsity = cal["sparsity"]
        print(f"  final sparsity={tgs_sparsity:.3f}")

        for method_name, args in methods.items():
            seed_results = []
            print(f"  [{method_name}]", end=" ", flush=True)
            for seed in SEEDS:
                set_seed(seed)
                if method_name in ("Fixed-schedule TGS", "Adaptive TGS"):
                    fn, fixed_ep = args
                    r = fn(data, nf, nc, seed, fixed_retire_epoch=fixed_ep)
                else:
                    fn, score_fn = args
                    r = fn(data, nf, nc, seed, tgs_sparsity, method_name, score_fn)
                # Find epoch when val first ≥ floor
                first_ep = None
                for ep, v in enumerate(r["val_curve"]):
                    if v >= acc_floor_val:
                        first_ep = ep
                        break
                r["epochs_to_target_acc"] = first_ep
                r["flops_at_target"] = r["flops_curve"][first_ep] if first_ep is not None else None
                r["flops_at_target_pct"] = (r["flops_curve"][first_ep] / dense_total_flops * 100
                                            if first_ep is not None else None)
                seed_results.append(r)
                print(".", end="", flush=True)
            print()

            val_m   = np.mean([r["best_val"] for r in seed_results])
            test_m  = np.mean([r["best_test"] for r in seed_results])
            sp_m    = np.mean([r["sparsity"]  for r in seed_results])
            fl_m    = np.mean([r["flops_used_pct"] for r in seed_results])
            apf_m   = np.mean([r["acc_per_flop"] for r in seed_results])
            ep_to_t = [r["epochs_to_target_acc"] for r in seed_results if r["epochs_to_target_acc"] is not None]
            fl_to_t = [r["flops_at_target_pct"] for r in seed_results if r["flops_at_target_pct"] is not None]

            ds_results[method_name] = {
                "val_mean": val_m, "val_std": np.std([r["best_val"] for r in seed_results]),
                "test_mean": test_m, "test_std": np.std([r["best_test"] for r in seed_results]),
                "sparsity_mean": sp_m,
                "flops_used_pct": fl_m,
                "acc_per_flop": apf_m,
                "epochs_to_floor": np.mean(ep_to_t) if ep_to_t else None,
                "flops_pct_to_floor": np.mean(fl_to_t) if fl_to_t else None,
                "seeds_hit_floor": len(ep_to_t),
            }
            print(f"    val={val_m:.3f}  test={test_m:.3f}  "
                  f"sp={sp_m:.2f}  FLOPs={fl_m:.1f}%  "
                  f"acc/GFLOP={apf_m:.2f}  "
                  f"epochs_to_floor={np.mean(ep_to_t):.0f}" if ep_to_t else
                  f"    val={val_m:.3f}  test={test_m:.3f}  NEVER hit floor")

        # Print summary table
        print(f"\n  ── {ds_name}: Accuracy-Constrained Summary ──")
        print(f"  Dense baseline: val={dense_val_mean:.3f}  accuracy floor: val≥{acc_floor_val:.3f}")
        print()
        print(f"  {'Method':<22} {'Val':>6} {'Test':>6} {'Sp':>5} "
              f"{'FLOPs%':>7} {'acc/GFLOP':>10} {'ep→floor':>9} {'FLOP%→floor':>12}")
        print(f"  {'─'*90}")
        print(f"  {'Dense GCN':<22} {dense_val_mean:>6.3f} {dense_test_mean:>6.3f} "
              f"{'0.00':>5} {'100.0':>7} {dense_test_mean/(dense_total_flops/1e9):>10.2f} "
              f"{'—':>9} {'100.0':>12}")
        for mn, rd in ds_results.items():
            if mn == "dense": continue
            efl = f"{rd['flops_pct_to_floor']:.1f}%" if rd["flops_pct_to_floor"] is not None else "N/A"
            etf = f"{rd['epochs_to_floor']:.0f}"      if rd["epochs_to_floor"]     is not None else "N/A"
            print(f"  {mn:<22} {rd['val_mean']:>6.3f} {rd['test_mean']:>6.3f} "
                  f"{rd['sparsity_mean']:>5.2f} {rd['flops_used_pct']:>7.1f} "
                  f"{rd['acc_per_flop']:>10.2f} {etf:>9} {efl:>12}")

        all_results[ds_name] = ds_results

    return all_results

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 2 — Budget-constrained race (50% FLOP budget)
# ─────────────────────────────────────────────────────────────────

def exp_budget_race(datasets):
    print("\n" + "="*72)
    print(f"EXPERIMENT 2 — BUDGET-CONSTRAINED RACE ({BUDGET_FRAC*100:.0f}% FLOP budget)")
    print("  Every method gets exactly 50% of dense cumulative message-passing FLOPs.")
    print("  Who achieves the highest accuracy within budget?")
    print("="*72)

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)
        m0 = data.edge_index.shape[1]
        dense_total = _flops(m0) * EPOCHS
        budget_flops = int(dense_total * BUDGET_FRAC)
        print(f"  Dense total FLOPs: {dense_total:,}   Budget: {budget_flops:,}")

        # Get TGS sparsity for matched baselines
        cal = train_tgs(data, nf, nc, seed=42)
        tgs_sp = cal["sparsity"]

        # For static methods: the budget is spent on a fixed sparse graph.
        # epochs_in_budget = budget_flops / flops_per_ep
        # If epochs_in_budget > EPOCHS, they get all EPOCHS; otherwise shorter run.

        method_configs = [
            ("Dense GCN",          "dense",  None),
            ("Random sparsify",    "static", score_random),
            ("Degree sparsify",    "static", score_degree),
            ("Eff. resistance",    "static", score_er),
            ("Fixed-sched TGS",   "tgs",    40),
            ("Adaptive TGS",       "tgs",    None),
        ]

        ds_results = {}

        for method_name, mode, arg in method_configs:
            seed_results = []
            print(f"  [{method_name}]", end=" ", flush=True)
            for seed in SEEDS:
                set_seed(seed)

                if mode == "dense":
                    # Dense: can only spend budget_flops.
                    # epochs_available = budget_flops / flops_per_ep
                    ep_budget = max(1, int(budget_flops / _flops(m0)))
                    r = train_dense(data, nf, nc, seed, epochs=min(ep_budget, EPOCHS))
                    r["method"] = method_name
                    r["budget_flops"] = budget_flops
                    r["flops_used"] = min(r["total_flops"], budget_flops)

                elif mode == "static":
                    # Static: compute epochs available within budget on sparse graph
                    n_keep = int(m0 * (1 - tgs_sp))
                    ep_flops = _flops(n_keep)
                    ep_budget = max(1, int(budget_flops / ep_flops))
                    r = train_static(data, nf, nc, seed, tgs_sp, method_name, arg,
                                     epochs=min(ep_budget, EPOCHS))
                    r["budget_flops"] = budget_flops
                    r["flops_used"] = r["total_flops"]

                else:  # TGS
                    # TGS: train for full EPOCHS but pass budget cap.
                    # Adaptation continues but gradient updates stop past budget.
                    r = train_tgs(data, nf, nc, seed,
                                  fixed_retire_epoch=arg,
                                  budget_cap_flops=budget_flops)
                    r["budget_flops"] = budget_flops
                    r["flops_used"] = min(r["total_flops"], budget_flops)

                seed_results.append(r)
                print(".", end="", flush=True)
            print()

            val_m  = np.mean([r["best_val"]  for r in seed_results])
            test_m = np.mean([r["best_test"] for r in seed_results])
            test_s = np.std([r["best_test"]  for r in seed_results])
            fl_u   = np.mean([r["flops_used"] for r in seed_results])
            sp_m   = np.mean([r["sparsity"]   for r in seed_results])

            ds_results[method_name] = {
                "val_mean": val_m, "test_mean": test_m, "test_std": test_s,
                "sparsity_mean": sp_m,
                "flops_used": fl_u,
                "flops_used_pct": fl_u / dense_total * 100,
                "acc_per_flop": test_m / max(fl_u / 1e9, 1e-12),
                "inf_ms": np.mean([r["inf_ms"] for r in seed_results]),
            }
            print(f"    val={val_m:.3f}  test={test_m:.3f}±{test_s:.3f}  "
                  f"sp={sp_m:.2f}  FLOPs_used={fl_u/dense_total*100:.1f}%")

        # Print race table
        print(f"\n  ── {ds_name}: Budget Race Results ({BUDGET_FRAC*100:.0f}% FLOPs) ──")
        print(f"  {'Method':<22} {'Val':>6} {'Test':>8} {'Sp':>5} "
              f"{'FLOPs%':>7} {'acc/GFLOP':>10} {'InfMs':>7} {'vs Dense':>9}")
        print(f"  {'─'*80}")
        dense_test = ds_results["Dense GCN"]["test_mean"] if "Dense GCN" in ds_results else 0
        for mn, rd in ds_results.items():
            diff = f"{(rd['test_mean'] - dense_test)*100:+.1f}pp" if mn != "Dense GCN" else "baseline"
            print(f"  {mn:<22} {rd['val_mean']:>6.3f} "
                  f"{rd['test_mean']:>6.3f}±{rd['test_std']:.3f} "
                  f"{rd['sparsity_mean']:>5.2f} {rd['flops_used_pct']:>7.1f} "
                  f"{rd['acc_per_flop']:>10.2f} {rd['inf_ms']:>7.3f} {diff:>9}")

        all_results[ds_name] = ds_results

    return all_results

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 3 — Accuracy-per-FLOPs Pareto frontier
# ─────────────────────────────────────────────────────────────────

def exp_pareto_frontier(datasets):
    """
    Sweep sparsity for static methods and timing for TGS.
    Plot Pareto: (FLOPs used, best test accuracy) for all methods.
    Headline metric: acc / (FLOPs / dense_FLOPs) — accuracy per compute unit.
    """
    print("\n" + "="*72)
    print("EXPERIMENT 3 — ACCURACY-PER-FLOP PARETO FRONTIER")
    print("  Headline metric: accuracy / (FLOPs_used / dense_FLOPs)")
    print("="*72)

    all_results = {}

    for ds_name in datasets[:2]:   # Cora + CiteSeer (time)
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)
        m0 = data.edge_index.shape[1]
        dense_total = _flops(m0) * EPOCHS

        frontier = {}

        # Dense reference
        r = train_dense(data, nf, nc, seed=42)
        frontier["Dense GCN"] = [(r["flops_used_pct"], r["best_test"],
                                   r["best_test"] / (r["total_flops"]/dense_total))]

        # Static degree — sweep sparsities
        print("  Static degree sweep:", end=" ", flush=True)
        deg_pts = []
        for sp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7]:
            accs = []
            for seed in [42, 43, 44]:
                set_seed(seed)
                r = train_static(data, nf, nc, seed, sp, "Degree", score_degree)
                accs.append(r["best_test"])
            acc_m = np.mean(accs)
            flops_pct = (1 - sp) * 100
            apf = acc_m / (flops_pct / 100)
            deg_pts.append((flops_pct, acc_m, apf))
            print(f"{sp:.0%}", end=" ", flush=True)
        frontier["Degree sparsify"] = deg_pts
        print()

        # Effective resistance — sweep sparsities
        print("  ER sweep:", end=" ", flush=True)
        er_pts = []
        for sp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7]:
            accs = []
            for seed in [42, 43, 44]:
                set_seed(seed)
                r = train_static(data, nf, nc, seed, sp, "ER", score_er)
                accs.append(r["best_test"])
            acc_m = np.mean(accs)
            flops_pct = (1 - sp) * 100
            apf = acc_m / (flops_pct / 100)
            er_pts.append((flops_pct, acc_m, apf))
            print(f"{sp:.0%}", end=" ", flush=True)
        frontier["Eff. resistance"] = er_pts
        print()

        # Adaptive TGS — sweep warmup timing (timing = when adaptation begins)
        print("  TGS timing sweep:", end=" ", flush=True)
        tgs_pts = []
        for warmup in [10, 20, 30, 40, 60, 80, 100]:
            cfg = dict(**TGS_CFG, warmup_steps=warmup)
            accs, flops = [], []
            for seed in [42, 43, 44]:
                set_seed(seed)
                m0_ = data.edge_index.shape[1]
                tg  = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
                est = GradientNormEstimator(m0_, DEVICE,
                        edge_index=data.edge_index, num_nodes=data.num_nodes,
                        alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
                model = TemporalGCN(nf, HIDDEN, nc, N_LAYERS, 0.5).to(DEVICE)
                opt   = torch.optim.Adam(list(model.parameters()) + [est.edge_weights],
                                         lr=0.01, weight_decay=5e-4)
                sched = AdaptiveRetirementScheduler(tg, **cfg)
                fc    = FLOPsCounter(m0_, N_LAYERS, HIDDEN)
                best_val = best_test = 0.0
                for ep in range(EPOCHS):
                    am = tg.active_mask
                    model.train()
                    logits = model(data.x, tg.edge_index, est.edge_weights[am])
                    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                    opt.zero_grad(); loss.backward()
                    est.update_influence(am); opt.step()
                    model.eval()
                    with torch.no_grad():
                        preds = model(data.x, tg.edge_index).argmax(-1)
                    v = (preds[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                    t_ = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                    sched.update_val_acc(v)
                    sched.step(est.influence_scores(am))
                    fc.record_step(tg.mt); tg.step()
                    if v > best_val: best_val, best_test = v, t_
                accs.append(best_test)
                flops.append(fc.total_flops / dense_total * 100)
            acc_m = np.mean(accs); fl_m = np.mean(flops)
            apf = acc_m / (fl_m / 100)
            tgs_pts.append((fl_m, acc_m, apf))
            print(f"w{warmup}", end=" ", flush=True)
        frontier["Adaptive TGS"] = tgs_pts
        print()

        # Print Pareto table
        print(f"\n  ── {ds_name}: Accuracy-per-FLOPs ──")
        print(f"  Headline metric: acc / (FLOPs_fraction)  — higher is better")
        print()

        # Best point per method
        for mn, pts in frontier.items():
            best = max(pts, key=lambda x: x[2])
            print(f"  {mn:<22}  FLOPs={best[0]:5.1f}%  acc={best[1]:.3f}  "
                  f"acc/compute={best[2]:.3f}  {'★ WINNER' if best[2] == max(max(p, key=lambda x:x[2]) for p in frontier.values())[2] else ''}")

        # Full Pareto table for TGS vs best static
        print(f"\n  TGS Pareto detail (warmup sweep):")
        print(f"  {'warmup':>8}  {'FLOPs%':>7}  {'Acc':>6}  {'Acc/Compute':>12}")
        warmups_list = [10, 20, 30, 40, 60, 80, 100]
        for i, pt in enumerate(frontier["Adaptive TGS"]):
            print(f"  {warmups_list[i]:>8}  {pt[0]:>7.1f}  {pt[1]:>6.3f}  {pt[2]:>12.3f}")

        all_results[ds_name] = {
            mn: [(float(a), float(b), float(c)) for a, b, c in pts]
            for mn, pts in frontier.items()
        }

    return all_results

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATASETS = ["Cora", "CiteSeer", "Texas", "Wisconsin"]

    print("\n" + "█"*72)
    print("  TGS ADAPTIVE COMPUTE-BUDGET EXPERIMENT")
    print("  'Can TGS make GNNs usable under real compute constraints?'")
    print("█"*72)
    print(f"  Datasets   : {DATASETS}")
    print(f"  Seeds      : {SEEDS}")
    print(f"  FLOP budget: {BUDGET_FRAC*100:.0f}% of dense")
    print(f"  Acc floor  : {ACC_FLOOR*100:.0f}% of dense val performance")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Output     : {OUT_DIR}/")

    all_output = {}

    # Experiment 1: Accuracy-constrained
    all_output["exp1_accuracy_constrained"] = exp_accuracy_constrained(DATASETS)

    # Experiment 2: Budget-constrained race
    all_output["exp2_budget_race"] = exp_budget_race(DATASETS)

    # Experiment 3: Pareto frontier
    all_output["exp3_pareto"] = exp_pareto_frontier(DATASETS)

    # Save
    out_path = os.path.join(OUT_DIR, "budget_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(all_output, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))

    print(f"\n\n{'█'*72}")
    print(f"  ALL EXPERIMENTS COMPLETE → {out_path}")
    print(f"{'█'*72}\n")
