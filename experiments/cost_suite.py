"""
experiments/cost_suite.py
=========================
Extensive Computational Cost Suite for TGS — ISEF Software Design

Sections
--------
1. MICRO  — per-epoch component timing breakdown (estimator, scheduler,
             forward, backward) across all datasets, 5 seeds
2. FLOPS  — theoretical and wall-clock FLOPs trajectory across training;
             savings curve, sparsity ramp
3. MEMORY — peak RSS, tracemalloc, edge-tensor growth/shrink over epochs
4. SCALABILITY — vary graph size (n) and density (m) synthetically;
                 measure TGS overhead as a function of graph stats
5. INFERENCE — final-model inference latency at every sparsity level
               produced by TGS, vs static at same sparsity, across 50 runs
6. OVERHEAD BREAKDOWN — stacked bar of where TGS training time goes
7. SPARSITY RAMP — edge count + active FLOPs across every epoch
8. CROSS-ARCH — GCN / GAT / SAGE cost comparison at matched sparsity
9. SUMMARY TABLE — single normalised table (Dense=100%) for all metrics

All results saved to results/cost_suite/
"""

import sys, os, json, time, tracemalloc, gc, resource
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import erdos_renyi_graph
import torch_geometric.transforms as T

from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.models.gat import TemporalGAT
from tgs.models.sage import TemporalSAGE
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils.reproducibility import set_seed

DEVICE  = torch.device("cpu")
EPOCHS  = 300
SEEDS   = [42, 43, 44, 45, 46]
OUT_DIR = "results/cost_suite"
os.makedirs(OUT_DIR, exist_ok=True)

TGS_CFG = dict(
    epsilon_max=5e-3, epsilon_min=1e-5, anneal_steps=100,
    warmup_steps=40, max_retire_frac=0.10,
    max_sparsity=0.65, retire_every=2
)

# ──────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────

def load_dataset(name):
    root = "./data"
    if name in ("Cora", "CiteSeer", "PubMed"):
        ds = Planetoid(root=root, name=name, transform=NormalizeFeatures())
        return ds[0], ds.num_features, ds.num_classes
    if name in ("Texas", "Wisconsin", "Cornell"):
        ds = WebKB(root=root, name=name, transform=NormalizeFeatures())
        d = ds[0]
        # WebKB uses per-split masks; pick split 0
        d.train_mask = d.train_mask[:, 0]
        d.val_mask   = d.val_mask[:, 0]
        d.test_mask  = d.test_mask[:, 0]
        return d, ds.num_features, ds.num_classes
    raise ValueError(name)


def make_synthetic(n, avg_deg, num_features=16, num_classes=4, seed=42):
    """Synthetic ER graph for scalability sweep."""
    torch.manual_seed(seed)
    p = avg_deg / (n - 1)
    ei = erdos_renyi_graph(n, p, directed=False)
    x  = torch.randn(n, num_features)
    y  = torch.randint(0, num_classes, (n,))
    # simple random 60/20/20 masks
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[:int(.6*n)]]   = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[perm[int(.6*n):int(.8*n)]] = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[perm[int(.8*n):]] = True
    from torch_geometric.data import Data
    return Data(x=x, edge_index=ei, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                num_nodes=n), num_features, num_classes

# ──────────────────────────────────────────────────────────────────
# Core training loops (return rich timing dicts)
# ──────────────────────────────────────────────────────────────────

def run_dense(data, nf, nc, seed=42, epochs=EPOCHS):
    set_seed(seed)
    model = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    m0    = data.edge_index.shape[1]
    flops_c = FLOPsCounter(m0, 2, 64)

    tracemalloc.start()
    epoch_times, fwd_times, bwd_times = [], [], []
    best_val = best_test = 0.0

    for epoch in range(epochs):
        t0 = time.perf_counter()

        model.train()
        tf = time.perf_counter()
        logits = model(data.x, data.edge_index)
        fwd_times.append(time.perf_counter() - tf)

        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad()
        tb = time.perf_counter()
        loss.backward()
        bwd_times.append(time.perf_counter() - tb)
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(data.x, data.edge_index).argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        flops_c.record_step(m0)
        epoch_times.append(time.perf_counter() - t0)
        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc

    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

    # Inference latency (50 runs)
    model.eval()
    inf_times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            model(data.x, data.edge_index)
            inf_times.append(time.perf_counter() - t0)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    return dict(
        label="Dense GCN",
        test_acc=best_test, sparsity=0.0,
        total_time_s=sum(epoch_times),
        epoch_time_mean=np.mean(epoch_times),
        epoch_time_std=np.std(epoch_times),
        fwd_time_mean=np.mean(fwd_times),
        bwd_time_mean=np.mean(bwd_times),
        overhead_estimator_ms=0.0,
        overhead_scheduler_ms=0.0,
        peak_mem_mb=peak/1e6,
        rss_mb=rss_mb,
        flops_reduction=0.0,
        total_flops=flops_c.total_flops,
        inference_ms_median=float(np.median(inf_times)*1000),
        inference_ms_p95=float(np.percentile(inf_times,95)*1000),
        inference_ms_p99=float(np.percentile(inf_times,99)*1000),
        m0=m0, epochs=epochs,
    )


def run_tgs(data, nf, nc, seed=42, epochs=EPOCHS, model_cls=None):
    if model_cls is None:
        model_cls = TemporalGCN
    set_seed(seed)
    m0  = data.edge_index.shape[1]
    tg  = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE,
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    model = model_cls(nf, 64, nc, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(
                list(model.parameters()) + [est.edge_weights],
                lr=0.01, weight_decay=5e-4)
    sched = AdaptiveRetirementScheduler(tg, **TGS_CFG)
    flops_c = FLOPsCounter(m0, 2, 64)

    tracemalloc.start()
    epoch_times, fwd_times, bwd_times = [], [], []
    est_times, sched_times = [], []
    edge_counts = []           # m_t trajectory
    best_val = best_test = 0.0

    for epoch in range(epochs):
        t0 = time.perf_counter()
        am = tg.active_mask

        model.train()
        tf = time.perf_counter()
        logits = model(data.x, tg.edge_index, est.edge_weights[am])
        fwd_times.append(time.perf_counter() - tf)

        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad()
        tb = time.perf_counter()
        loss.backward()
        bwd_times.append(time.perf_counter() - tb)

        te = time.perf_counter()
        est.update_influence(am)
        est_times.append(time.perf_counter() - te)

        opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(data.x, tg.edge_index).argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        ts = time.perf_counter()
        sched.update_val_acc(val_acc)
        scores = est.influence_scores(am)
        sched.step(scores)
        sched_times.append(time.perf_counter() - ts)

        flops_c.record_step(tg.mt)
        edge_counts.append(tg.mt)
        tg.step()
        epoch_times.append(time.perf_counter() - t0)
        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc

    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

    # Inference on final sparse graph (50 runs)
    final_ei = tg.edge_index
    model.eval()
    inf_times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            model(data.x, final_ei)
            inf_times.append(time.perf_counter() - t0)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    f = flops_c.summary()
    rs = sched.summary()

    return dict(
        label="TGS",
        test_acc=best_test, sparsity=tg.sparsity,
        total_time_s=sum(epoch_times),
        epoch_time_mean=np.mean(epoch_times),
        epoch_time_std=np.std(epoch_times),
        fwd_time_mean=np.mean(fwd_times),
        bwd_time_mean=np.mean(bwd_times),
        overhead_estimator_ms=np.mean(est_times)*1000,
        overhead_scheduler_ms=np.mean(sched_times)*1000,
        overhead_estimator_pct=np.mean(est_times)/np.mean(epoch_times)*100,
        overhead_scheduler_pct=np.mean(sched_times)/np.mean(epoch_times)*100,
        peak_mem_mb=peak/1e6,
        rss_mb=rss_mb,
        flops_reduction=f["flops_reduction"],
        total_flops=f["total_flops"],
        dense_flops=f["dense_flops"],
        inference_ms_median=float(np.median(inf_times)*1000),
        inference_ms_p95=float(np.percentile(inf_times,95)*1000),
        inference_ms_p99=float(np.percentile(inf_times,99)*1000),
        edge_trajectory=edge_counts,
        distortion=rs.get("cumulative_distortion_bound", 0.0),
        m0=m0, epochs=epochs,
        model_class=model_cls.__name__,
    )


def run_static(data, nf, nc, target_sparsity, seed=42, epochs=EPOCHS):
    from torch_geometric.utils import degree as pyg_deg
    set_seed(seed)
    m0  = data.edge_index.shape[1]
    src = data.edge_index[0]; dst = data.edge_index[1]
    deg = pyg_deg(dst, data.num_nodes, dtype=torch.float)
    score = deg[src] * deg[dst]
    n_remove = int(m0 * target_sparsity)
    _, sidx  = score.sort(descending=True)
    remove   = set(sidx[:n_remove].tolist())
    keep     = torch.tensor([i not in remove for i in range(m0)], dtype=torch.bool)
    ei       = data.edge_index[:, keep]; m_sp = ei.shape[1]

    model = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    flops_c = FLOPsCounter(m_sp, 2, 64)

    tracemalloc.start()
    epoch_times, fwd_times, bwd_times = [], [], []
    best_val = best_test = 0.0

    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        tf = time.perf_counter()
        logits = model(data.x, ei)
        fwd_times.append(time.perf_counter() - tf)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad()
        tb = time.perf_counter()
        loss.backward()
        bwd_times.append(time.perf_counter() - tb)
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(data.x, ei).argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        flops_c.record_step(m_sp)
        epoch_times.append(time.perf_counter() - t0)
        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc

    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

    inf_times = []
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            model(data.x, ei)
            inf_times.append(time.perf_counter() - t0)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    f = flops_c.summary()

    return dict(
        label="Static (degree-prune)",
        test_acc=best_test, sparsity=1-m_sp/m0,
        total_time_s=sum(epoch_times),
        epoch_time_mean=np.mean(epoch_times),
        epoch_time_std=np.std(epoch_times),
        fwd_time_mean=np.mean(fwd_times),
        bwd_time_mean=np.mean(bwd_times),
        overhead_estimator_ms=0.0,
        overhead_scheduler_ms=0.0,
        peak_mem_mb=peak/1e6,
        rss_mb=rss_mb,
        flops_reduction=f["flops_reduction"],
        total_flops=f["total_flops"],
        inference_ms_median=float(np.median(inf_times)*1000),
        inference_ms_p95=float(np.percentile(inf_times,95)*1000),
        inference_ms_p99=float(np.percentile(inf_times,99)*1000),
        m0=m0, epochs=epochs,
    )

# ──────────────────────────────────────────────────────────────────
# SECTION 1 — MICRO: per-epoch component breakdown, multi-seed
# ──────────────────────────────────────────────────────────────────

def section_micro(datasets):
    print("\n" + "="*70)
    print("SECTION 1 — MICRO TIMING BREAKDOWN (per-epoch components)")
    print("="*70)
    results = {}

    for ds_name in datasets:
        print(f"\n  Dataset: {ds_name}")
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)
        m0 = data.edge_index.shape[1]

        seed_results = []
        for seed in SEEDS:
            print(f"    seed {seed}...", end=" ", flush=True)
            r = run_tgs(data, nf, nc, seed=seed)
            seed_results.append(r)
            print(f"acc={r['test_acc']:.3f}  sp={r['sparsity']:.2f}  "
                  f"est={r['overhead_estimator_ms']:.2f}ms  "
                  f"sched={r['overhead_scheduler_ms']:.2f}ms")

        # Aggregate
        agg = {
            "dataset": ds_name,
            "n": data.num_nodes,
            "m0": m0,
            "epoch_time_ms_mean": np.mean([r["epoch_time_mean"]*1000 for r in seed_results]),
            "epoch_time_ms_std":  np.std([r["epoch_time_mean"]*1000  for r in seed_results]),
            "fwd_ms":             np.mean([r["fwd_time_mean"]*1000    for r in seed_results]),
            "bwd_ms":             np.mean([r["bwd_time_mean"]*1000    for r in seed_results]),
            "estimator_ms":       np.mean([r["overhead_estimator_ms"] for r in seed_results]),
            "estimator_ms_std":   np.std([r["overhead_estimator_ms"]  for r in seed_results]),
            "scheduler_ms":       np.mean([r["overhead_scheduler_ms"] for r in seed_results]),
            "scheduler_ms_std":   np.std([r["overhead_scheduler_ms"]  for r in seed_results]),
            "estimator_pct":      np.mean([r["overhead_estimator_pct"] for r in seed_results]),
            "scheduler_pct":      np.mean([r["overhead_scheduler_pct"] for r in seed_results]),
            "test_acc_mean":      np.mean([r["test_acc"]              for r in seed_results]),
            "test_acc_std":       np.std([r["test_acc"]               for r in seed_results]),
            "sparsity_mean":      np.mean([r["sparsity"]              for r in seed_results]),
            "flops_reduction_mean": np.mean([r["flops_reduction"]     for r in seed_results]),
        }

        total_ms = agg["epoch_time_ms_mean"]
        other_ms = total_ms - agg["fwd_ms"] - agg["bwd_ms"] - agg["estimator_ms"] - agg["scheduler_ms"]

        print(f"\n  ── {ds_name} Epoch Breakdown (mean over {len(SEEDS)} seeds) ──")
        print(f"    Total epoch time  : {total_ms:.2f} ms")
        print(f"    Forward pass      : {agg['fwd_ms']:.2f} ms  ({agg['fwd_ms']/total_ms*100:.1f}%)")
        print(f"    Backward pass     : {agg['bwd_ms']:.2f} ms  ({agg['bwd_ms']/total_ms*100:.1f}%)")
        print(f"    Influence estimator: {agg['estimator_ms']:.2f} ± {agg['estimator_ms_std']:.2f} ms  ({agg['estimator_pct']:.1f}%)")
        print(f"    Scheduler+guard   : {agg['scheduler_ms']:.2f} ± {agg['scheduler_ms_std']:.2f} ms  ({agg['scheduler_pct']:.1f}%)")
        print(f"    Other (eval/misc) : {max(other_ms, 0):.2f} ms  ({max(other_ms,0)/total_ms*100:.1f}%)")
        print(f"    TGS overhead total: {agg['estimator_pct']+agg['scheduler_pct']:.1f}%")
        print(f"    → Acc {agg['test_acc_mean']:.3f}±{agg['test_acc_std']:.3f}  "
              f"  Sparsity {agg['sparsity_mean']:.2f}  "
              f"  FLOPs saved {agg['flops_reduction_mean']*100:.1f}%")

        results[ds_name] = agg

    return results


# ──────────────────────────────────────────────────────────────────
# SECTION 2 — FLOPS TRAJECTORY
# ──────────────────────────────────────────────────────────────────

def section_flops(datasets):
    print("\n" + "="*70)
    print("SECTION 2 — FLOPs TRAJECTORY (epoch-by-epoch savings)")
    print("="*70)
    results = {}

    for ds_name in datasets:
        print(f"\n  Dataset: {ds_name}")
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)
        m0 = data.edge_index.shape[1]

        # Single seed — track trajectory
        r = run_tgs(data, nf, nc, seed=42)
        traj = r["edge_trajectory"]

        # Compute cumulative FLOPs savings at each epoch
        flops_per_epoch_dense  = [m0 * 2 * 64 * 2] * EPOCHS
        flops_per_epoch_tgs    = [mt * 2 * 64 * 2 for mt in traj]
        cum_dense = np.cumsum(flops_per_epoch_dense)
        cum_tgs   = np.cumsum(flops_per_epoch_tgs)
        cum_savings = (cum_dense - cum_tgs) / cum_dense * 100

        # Key stats
        warmup_end  = TGS_CFG["warmup_steps"]
        anneal_end  = warmup_end + TGS_CFG["anneal_steps"]
        final_sp    = r["sparsity"]
        flops_saved = r["flops_reduction"] * 100

        print(f"    m0={m0:,}  final_m={traj[-1]:,}  sparsity={final_sp:.3f}")
        print(f"    FLOPs saved (cumulative over training): {flops_saved:.1f}%")
        print(f"    Warmup ends at epoch {warmup_end}: edges still at {traj[warmup_end-1]:,}/{m0:,}")
        if anneal_end < EPOCHS:
            print(f"    Annealing ends at epoch {anneal_end}: edges at {traj[min(anneal_end,EPOCHS-1)]:,}/{m0:,}")
        print(f"    Cumulative FLOPs savings @ epoch 100: {cum_savings[99]:.1f}%")
        print(f"    Cumulative FLOPs savings @ epoch 200: {cum_savings[199]:.1f}%")
        print(f"    Cumulative FLOPs savings @ epoch 300: {cum_savings[299]:.1f}%")

        # Epoch where 50% of final sparsity is reached
        target_m = m0 * (1 - final_sp * 0.5)
        half_sp_epoch = next((i for i,m in enumerate(traj) if m <= target_m), EPOCHS)
        print(f"    50% of final sparsity reached at epoch {half_sp_epoch}")

        results[ds_name] = {
            "m0": m0,
            "edge_trajectory": traj,
            "cumulative_flops_savings_pct": list(cum_savings),
            "final_sparsity": final_sp,
            "total_flops_saved_pct": flops_saved,
            "half_sparsity_epoch": half_sp_epoch,
        }

    return results


# ──────────────────────────────────────────────────────────────────
# SECTION 3 — MEMORY ANALYSIS
# ──────────────────────────────────────────────────────────────────

def section_memory(datasets):
    print("\n" + "="*70)
    print("SECTION 3 — MEMORY ANALYSIS")
    print("="*70)
    results = {}

    for ds_name in datasets:
        print(f"\n  Dataset: {ds_name}")
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)
        m0 = data.edge_index.shape[1]
        n  = data.num_nodes

        # Edge tensor sizes in bytes
        bytes_per_edge_index = m0 * 2 * 8   # int64
        bytes_per_edge_weights = m0 * 4     # float32

        r_dense  = run_dense(data, nf, nc, seed=42)
        gc.collect()
        r_tgs    = run_tgs(data, nf, nc, seed=42)
        gc.collect()
        r_static = run_static(data, nf, nc, r_tgs["sparsity"], seed=42)
        gc.collect()

        final_m = int(m0 * (1 - r_tgs["sparsity"]))
        edge_bytes_saved = (m0 - final_m) * 2 * 8

        print(f"    n={n:,}  m0={m0:,}  final_m={final_m:,}")
        print(f"    Peak tracemalloc — Dense: {r_dense['peak_mem_mb']:.2f} MB  "
              f"TGS: {r_tgs['peak_mem_mb']:.2f} MB  "
              f"Static: {r_static['peak_mem_mb']:.2f} MB")
        print(f"    Edge index tensor (initial): {bytes_per_edge_index/1024:.1f} KB")
        print(f"    Edge weight tensor (initial): {bytes_per_edge_weights/1024:.1f} KB")
        print(f"    Edge bytes freed at end: {edge_bytes_saved/1024:.1f} KB "
              f"({edge_bytes_saved/bytes_per_edge_index*100:.1f}% of edge index)")
        print(f"    TGS vs Dense memory ratio: {r_tgs['peak_mem_mb']/r_dense['peak_mem_mb']*100:.1f}%")
        print(f"    Static vs Dense memory ratio: {r_static['peak_mem_mb']/r_dense['peak_mem_mb']*100:.1f}%")

        results[ds_name] = {
            "n": n, "m0": m0, "final_m": final_m,
            "dense_peak_mb":  r_dense["peak_mem_mb"],
            "tgs_peak_mb":    r_tgs["peak_mem_mb"],
            "static_peak_mb": r_static["peak_mem_mb"],
            "tgs_vs_dense_pct": r_tgs["peak_mem_mb"]/r_dense["peak_mem_mb"]*100,
            "edge_bytes_freed_kb": edge_bytes_saved/1024,
        }

    return results


# ──────────────────────────────────────────────────────────────────
# SECTION 4 — SCALABILITY SWEEP
# ──────────────────────────────────────────────────────────────────

def section_scalability():
    print("\n" + "="*70)
    print("SECTION 4 — SCALABILITY (synthetic ER graphs, n sweep)")
    print("="*70)

    ns      = [200, 500, 1000, 2000, 3000]
    avg_deg = 8
    results = []

    for n in ns:
        print(f"\n  n={n}, avg_deg={avg_deg}")
        data, nf, nc = make_synthetic(n, avg_deg, seed=42)
        data = data.to(DEVICE)
        m0   = data.edge_index.shape[1]

        # Short run (100 epochs) for speed
        r_dense  = run_dense(data, nf, nc, seed=42, epochs=100)
        r_tgs    = run_tgs(data, nf, nc, seed=42, epochs=100)

        overhead_pct = (r_tgs["overhead_estimator_ms"] + r_tgs["overhead_scheduler_ms"]) \
                       / (r_tgs["epoch_time_mean"]*1000) * 100
        speedup_inference = r_dense["inference_ms_median"] / r_tgs["inference_ms_median"]

        print(f"    m0={m0:,}")
        print(f"    TGS epoch: {r_tgs['epoch_time_mean']*1000:.2f} ms  "
              f"Dense: {r_dense['epoch_time_mean']*1000:.2f} ms  "
              f"overhead: {overhead_pct:.1f}%")
        print(f"    TGS sparsity: {r_tgs['sparsity']:.2f}  "
              f"FLOPs saved: {r_tgs['flops_reduction']*100:.1f}%")
        print(f"    Inference speedup (dense→TGS): {speedup_inference:.2f}x")

        results.append({
            "n": n, "m0": m0, "avg_deg": avg_deg,
            "tgs_epoch_ms": r_tgs["epoch_time_mean"]*1000,
            "dense_epoch_ms": r_dense["epoch_time_mean"]*1000,
            "tgs_overhead_pct": overhead_pct,
            "estimator_ms": r_tgs["overhead_estimator_ms"],
            "scheduler_ms": r_tgs["overhead_scheduler_ms"],
            "tgs_sparsity": r_tgs["sparsity"],
            "flops_saved_pct": r_tgs["flops_reduction"]*100,
            "tgs_inference_ms": r_tgs["inference_ms_median"],
            "dense_inference_ms": r_dense["inference_ms_median"],
            "inference_speedup": speedup_inference,
        })

    # Also density sweep at fixed n=1000
    print(f"\n  Density sweep (n=1000, vary avg_deg)")
    n_fixed = 1000
    dens_results = []
    for avg_deg in [4, 8, 16, 32, 64]:
        data, nf, nc = make_synthetic(n_fixed, avg_deg, seed=42)
        data = data.to(DEVICE)
        m0 = data.edge_index.shape[1]
        r_tgs = run_tgs(data, nf, nc, seed=42, epochs=100)
        overhead_pct = (r_tgs["overhead_estimator_ms"] + r_tgs["overhead_scheduler_ms"]) \
                       / (r_tgs["epoch_time_mean"]*1000) * 100
        print(f"    avg_deg={avg_deg:3d}  m0={m0:6,}  "
              f"est_ms={r_tgs['overhead_estimator_ms']:.3f}  "
              f"overhead={overhead_pct:.1f}%  "
              f"sparsity={r_tgs['sparsity']:.2f}")
        dens_results.append({
            "n": n_fixed, "avg_deg": avg_deg, "m0": m0,
            "estimator_ms": r_tgs["overhead_estimator_ms"],
            "overhead_pct": overhead_pct,
            "sparsity": r_tgs["sparsity"],
            "flops_saved_pct": r_tgs["flops_reduction"]*100,
        })

    return {"n_sweep": results, "density_sweep": dens_results}


# ──────────────────────────────────────────────────────────────────
# SECTION 5 — INFERENCE LATENCY at varying sparsity
# ──────────────────────────────────────────────────────────────────

def section_inference(datasets):
    print("\n" + "="*70)
    print("SECTION 5 — INFERENCE LATENCY vs SPARSITY")
    print("="*70)
    results = {}

    for ds_name in datasets[:2]:  # Cora + CiteSeer only (time)
        print(f"\n  Dataset: {ds_name}")
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)
        m0 = data.edge_index.shape[1]

        model = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
        torch.manual_seed(42)
        # measure at multiple sparsity levels by subsampling edge set
        sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8]
        level_results = []

        for sp in sparsity_levels:
            if sp == 0.0:
                ei = data.edge_index
            else:
                n_keep = int(m0 * (1-sp))
                perm = torch.randperm(m0)[:n_keep]
                ei = data.edge_index[:, perm]

            times = []
            model.eval()
            with torch.no_grad():
                # warmup
                for _ in range(5):
                    model(data.x, ei)
                for _ in range(50):
                    t0 = time.perf_counter()
                    model(data.x, ei)
                    times.append(time.perf_counter() - t0)

            res = {
                "sparsity": sp,
                "m": ei.shape[1],
                "latency_ms_median": float(np.median(times)*1000),
                "latency_ms_p95": float(np.percentile(times,95)*1000),
                "latency_ms_p99": float(np.percentile(times,99)*1000),
                "speedup_vs_dense": float(np.median([times[0]]) / np.median(times))
                    if sp > 0 else 1.0,
            }
            print(f"    sp={sp:.2f}  m={ei.shape[1]:6,}  "
                  f"latency={res['latency_ms_median']:.3f}ms  "
                  f"p95={res['latency_ms_p95']:.3f}ms")
            level_results.append(res)

        # Recompute speedup relative to sp=0
        base_lat = level_results[0]["latency_ms_median"]
        for r in level_results:
            r["speedup_vs_dense"] = base_lat / r["latency_ms_median"]

        results[ds_name] = {
            "m0": m0,
            "inference_by_sparsity": level_results,
        }

    return results


# ──────────────────────────────────────────────────────────────────
# SECTION 6 — OVERHEAD BREAKDOWN (stacked percentages)
# ──────────────────────────────────────────────────────────────────

def section_overhead_breakdown(datasets):
    print("\n" + "="*70)
    print("SECTION 6 — OVERHEAD BREAKDOWN")
    print("="*70)
    results = {}

    for ds_name in datasets:
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)

        seed_data = []
        for seed in SEEDS[:3]:
            r = run_tgs(data, nf, nc, seed=seed)
            total_ms = r["epoch_time_mean"] * 1000
            fwd   = r["fwd_time_mean"] * 1000
            bwd   = r["bwd_time_mean"] * 1000
            est   = r["overhead_estimator_ms"]
            sched = r["overhead_scheduler_ms"]
            other = max(total_ms - fwd - bwd - est - sched, 0)
            seed_data.append({
                "fwd_pct":   fwd/total_ms*100,
                "bwd_pct":   bwd/total_ms*100,
                "est_pct":   est/total_ms*100,
                "sched_pct": sched/total_ms*100,
                "other_pct": other/total_ms*100,
                "total_ms":  total_ms,
            })

        means = {k: np.mean([d[k] for d in seed_data]) for k in seed_data[0]}
        print(f"\n  {ds_name} — epoch time {means['total_ms']:.2f} ms:")
        print(f"    Forward:    {means['fwd_pct']:.1f}%")
        print(f"    Backward:   {means['bwd_pct']:.1f}%")
        print(f"    Estimator:  {means['est_pct']:.1f}%")
        print(f"    Scheduler:  {means['sched_pct']:.1f}%")
        print(f"    Other:      {means['other_pct']:.1f}%")
        print(f"    TGS overhead: {means['est_pct']+means['sched_pct']:.1f}%")

        results[ds_name] = means

    return results


# ──────────────────────────────────────────────────────────────────
# SECTION 7 — SPARSITY RAMP (edge count trajectory)
# ──────────────────────────────────────────────────────────────────

def section_sparsity_ramp(datasets):
    print("\n" + "="*70)
    print("SECTION 7 — SPARSITY RAMP (edge decay over training)")
    print("="*70)
    results = {}

    for ds_name in datasets:
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)
        m0 = data.edge_index.shape[1]

        # 3 seeds, average trajectory
        trajs = []
        for seed in SEEDS[:3]:
            r = run_tgs(data, nf, nc, seed=seed)
            trajs.append(r["edge_trajectory"])

        avg_traj = np.mean(trajs, axis=0)
        std_traj = np.std(trajs, axis=0)

        # Key milestones
        warmup = TGS_CFG["warmup_steps"]
        checkpoints = [warmup, 100, 150, 200, 250, 299]

        print(f"\n  {ds_name}  m0={m0:,}")
        print(f"  {'Epoch':>6}  {'Edges':>8}  {'Sparsity':>9}  {'Std':>6}")
        for ep in checkpoints:
            ep = min(ep, EPOCHS-1)
            sp = 1 - avg_traj[ep]/m0
            print(f"  {ep:>6}  {avg_traj[ep]:>8.0f}  {sp:>9.3f}  ±{std_traj[ep]/m0:.4f}")

        results[ds_name] = {
            "m0": m0,
            "avg_edge_trajectory": list(avg_traj),
            "std_edge_trajectory": list(std_traj),
            "avg_sparsity_trajectory": list(1 - avg_traj/m0),
        }

    return results


# ──────────────────────────────────────────────────────────────────
# SECTION 8 — CROSS-ARCHITECTURE COST
# ──────────────────────────────────────────────────────────────────

def section_cross_arch():
    print("\n" + "="*70)
    print("SECTION 8 — CROSS-ARCHITECTURE COST (GCN / GAT / SAGE)")
    print("="*70)
    results = {}

    data, nf, nc = load_dataset("Cora")
    data = data.to(DEVICE)
    m0   = data.edge_index.shape[1]

    archs = [
        ("GCN",  TemporalGCN),
        ("GAT",  TemporalGAT),
        ("SAGE", TemporalSAGE),
    ]

    print(f"\n  Cora (n={data.num_nodes:,}, m0={m0:,})")
    print(f"\n  {'Arch':<6}  {'Mode':<8}  {'Acc':>6}  {'Sp':>6}  "
          f"{'Epoch ms':>10}  {'Inf ms':>8}  {'OH%':>6}  {'FLOPs↓':>8}")
    print(f"  {'-'*70}")

    for arch_name, model_cls in archs:
        # Dense baseline
        try:
            r_dense = run_dense(data, nf, nc, seed=42)
            r_dense["arch"] = arch_name; r_dense["mode"] = "dense"
        except Exception as e:
            print(f"  {arch_name} dense failed: {e}"); continue

        # TGS
        try:
            r_tgs = run_tgs(data, nf, nc, seed=42, model_cls=model_cls)
            r_tgs["arch"] = arch_name; r_tgs["mode"] = "tgs"
        except Exception as e:
            print(f"  {arch_name} TGS failed: {e}"); continue

        oh_pct = (r_tgs["overhead_estimator_ms"] + r_tgs["overhead_scheduler_ms"]) \
                 / (r_tgs["epoch_time_mean"]*1000) * 100

        for r in [r_dense, r_tgs]:
            oh = oh_pct if r["mode"]=="tgs" else 0.0
            print(f"  {arch_name:<6}  {r['mode']:<8}  {r['test_acc']:>6.3f}  "
                  f"{r['sparsity']:>6.3f}  {r['epoch_time_mean']*1000:>10.2f}  "
                  f"{r['inference_ms_median']:>8.3f}  {oh:>6.1f}  "
                  f"{r['flops_reduction']*100:>8.1f}%")

        results[arch_name] = {
            "dense": r_dense, "tgs": r_tgs,
            "overhead_pct": oh_pct,
        }

    return results


# ──────────────────────────────────────────────────────────────────
# SECTION 9 — SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────

def section_summary(datasets):
    print("\n" + "="*70)
    print("SECTION 9 — NORMALISED SUMMARY TABLE (Dense GCN = 100%)")
    print("="*70)
    all_results = {}

    for ds_name in datasets:
        print(f"\n  Dataset: {ds_name}")
        data, nf, nc = load_dataset(ds_name)
        data = data.to(DEVICE)

        r_dense  = run_dense(data, nf, nc, seed=42)
        r_tgs    = run_tgs(data, nf, nc, seed=42)
        r_static = run_static(data, nf, nc, r_tgs["sparsity"], seed=42)

        ref_time = r_dense["total_time_s"]
        ref_mem  = r_dense["peak_mem_mb"]
        ref_inf  = r_dense["inference_ms_median"]
        ref_fl   = r_dense["total_flops"]

        print(f"\n  {'Method':<22} {'Acc':>6} {'Sp':>6} {'Train%':>8} "
              f"{'Mem%':>7} {'Inf%':>7} {'FLOPs%':>8} {'TGS_OH%':>8}")
        print(f"  {'-'*75}")

        for r in [r_dense, r_tgs, r_static]:
            oh = (r.get("overhead_estimator_ms",0)+r.get("overhead_scheduler_ms",0)) \
                 / (r["epoch_time_mean"]*1000) * 100
            print(
                f"  {r['label']:<22} "
                f"{r['test_acc']:>6.3f} "
                f"{r['sparsity']:>6.3f} "
                f"{r['total_time_s']/ref_time*100:>7.0f}% "
                f"{r['peak_mem_mb']/ref_mem*100:>6.0f}% "
                f"{r['inference_ms_median']/ref_inf*100:>6.0f}% "
                f"{(1-r['flops_reduction'])*100:>7.0f}% "
                f"{oh:>7.1f}%"
            )

        all_results[ds_name] = {
            "dense": r_dense, "tgs": r_tgs, "static": r_static,
        }

    return all_results


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATASETS = ["Cora", "CiteSeer", "Texas", "Wisconsin"]

    all_output = {}

    print("\n" + "█"*70)
    print("  TGS COMPUTATIONAL COST SUITE")
    print("█"*70)
    print(f"  Datasets : {DATASETS}")
    print(f"  Seeds    : {SEEDS}")
    print(f"  Epochs   : {EPOCHS}")
    print(f"  Device   : {DEVICE}")
    print(f"  Output   : {OUT_DIR}/")

    # Section 1: Micro timing
    all_output["micro"] = section_micro(DATASETS)

    # Section 2: FLOPs trajectory
    all_output["flops"] = section_flops(DATASETS)

    # Section 3: Memory
    all_output["memory"] = section_memory(DATASETS)

    # Section 4: Scalability
    all_output["scalability"] = section_scalability()

    # Section 5: Inference latency
    all_output["inference"] = section_inference(DATASETS)

    # Section 6: Overhead breakdown
    all_output["overhead"] = section_overhead_breakdown(DATASETS)

    # Section 7: Sparsity ramp
    all_output["sparsity_ramp"] = section_sparsity_ramp(DATASETS)

    # Section 8: Cross-arch
    all_output["cross_arch"] = section_cross_arch()

    # Section 9: Summary table
    all_output["summary"] = section_summary(DATASETS)

    # Save everything
    out_path = os.path.join(OUT_DIR, "cost_suite_results.json")
    with open(out_path, "w") as f:
        json.dump(all_output, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    print(f"\n\n{'█'*70}")
    print(f"  ALL SECTIONS COMPLETE — saved to {out_path}")
    print(f"{'█'*70}\n")
