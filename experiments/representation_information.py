"""
experiments/representation_information.py

The "deadly" experiment: does representation maturity predict
when pruning becomes safe?

Protocol:
  1. Train dense GCN on Cora/CiteSeer/PubMed.
  2. At epochs {0,10,20,30,40,60,80,120}, freeze H_t.
  3. Fit a linear probe on H_t -> labels (no graph, no GNN).
  4. Record probe accuracy = representation maturity signal.
  5. At each epoch t, also retire edges (start from t, train to 300).
  6. Record final TGS accuracy for each warmup value.
  7. Compute correlation: probe_acc(t) <-> TGS_gain(t).

Causal chain we are testing:
  graph structure
       |
       v
  representation formation (measured by probe)
       |
       v
  edges become less necessary (safe to retire)
       |
       v
  sparsification without information loss

If probe_acc(t) strongly predicts TGS_gain(t) across warmup values
AND across datasets, the causal chain is established.
"""

import sys, os, json, time
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.WARNING)

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree
from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils.reproducibility import set_seed

DEVICE  = torch.device("cpu")
EPOCHS  = 300
# Pre-registered warmup grid (chosen before seeing results)
WARMUP_GRID = [0, 10, 20, 30, 40, 60, 80, 120]
SEEDS   = [42, 123, 456]


# ── Linear probe ─────────────────────────────────────────────────────────────

def linear_probe_accuracy(H: torch.Tensor, y: torch.Tensor,
                          train_mask: torch.Tensor, test_mask: torch.Tensor,
                          n_epochs: int = 200) -> float:
    """
    Fit a linear classifier on frozen representations H.
    No GNN, no graph — pure representation quality measurement.
    Returns test accuracy.
    """
    d = H.shape[1]; nc = int(y.max().item()) + 1
    probe = nn.Linear(d, nc).to(DEVICE)
    opt   = torch.optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)
    H_det = H.detach()

    best = 0.0
    for _ in range(n_epochs):
        probe.train()
        loss = F.cross_entropy(probe(H_det[train_mask]), y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            preds = probe(H_det[test_mask]).argmax(-1)
        acc = (preds == y[test_mask]).float().mean().item()
        if acc > best: best = acc
    return best


def representation_similarity(H1: torch.Tensor, H2: torch.Tensor) -> float:
    """CKA (centered kernel alignment) between two representation matrices."""
    def centering(K):
        n = K.shape[0]
        unit = torch.ones(n, n, device=K.device) / n
        return K - unit @ K - K @ unit + unit @ K @ unit

    H1n = F.normalize(H1.detach(), dim=1)
    H2n = F.normalize(H2.detach(), dim=1)
    K1  = H1n @ H1n.T
    K2  = H2n @ H2n.T
    K1c = centering(K1)
    K2c = centering(K2)
    hsic = (K1c * K2c).sum()
    n1   = (K1c * K1c).sum().sqrt()
    n2   = (K2c * K2c).sum().sqrt()
    return (hsic / (n1 * n2 + 1e-8)).item()


# ── Main experiment ───────────────────────────────────────────────────────────

def run_dataset(ds_name: str):
    dataset = Planetoid(root="./data", name=ds_name, transform=NormalizeFeatures())
    data    = dataset[0].to(DEVICE)
    nf, nc  = dataset.num_features, dataset.num_classes
    m0      = data.edge_index.shape[1]

    print(f"\n{'='*65}")
    print(f"Dataset: {ds_name}  (n={data.num_nodes}, m={m0})")
    print(f"{'='*65}")

    # Static baseline accuracy (fixed across seeds for this dataset)
    src_np, dst_np = data.edge_index[0].numpy(), data.edge_index[1].numpy()
    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float).numpy()
    er  = 1.0 / deg[src_np].clip(1) + 1.0 / deg[dst_np].clip(1)
    score = torch.from_numpy(deg[src_np] * deg[dst_np]).float()
    score[torch.from_numpy(er) >= torch.quantile(torch.from_numpy(er), 0.90)] = -1.0

    # Run across seeds
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n  Seed {seed}:")
        seed_results = []

        # Step 1: Train dense model fully, capture H_t at each warmup epoch
        # and get final H_T for CKA reference
        model_full = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
        opt_full   = torch.optim.Adam(model_full.parameters(), lr=0.01, weight_decay=5e-4)
        snapshots  = {}  # epoch -> H_t

        for epoch in range(max(WARMUP_GRID) + 1):
            model_full.train()
            F.cross_entropy(model_full(data.x, data.edge_index)[data.train_mask],
                            data.y[data.train_mask]).backward()
            opt_full.step(); opt_full.zero_grad()
            if epoch in WARMUP_GRID:
                model_full.eval()
                with torch.no_grad():
                    H_t = model_full(data.x, data.edge_index)
                snapshots[epoch] = H_t.clone()

        # Continue to convergence for final reference H_T
        for epoch in range(max(WARMUP_GRID) + 1, EPOCHS):
            model_full.train()
            F.cross_entropy(model_full(data.x, data.edge_index)[data.train_mask],
                            data.y[data.train_mask]).backward()
            opt_full.step(); opt_full.zero_grad()

        model_full.eval()
        with torch.no_grad():
            H_final = model_full(data.x, data.edge_index)
        dense_preds = H_final.argmax(-1)
        dense_acc   = (dense_preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        print(f"    Dense final acc: {dense_acc:.4f}")
        print(f"    {'Warmup':>7} {'Probe':>7} {'CKA':>6} {'TGS':>7} {'Static':>8} {'TGS-Stat':>9}")
        print(f"    {'-'*52}")

        for warmup in WARMUP_GRID:
            # Probe accuracy at this warmup epoch
            H_t        = snapshots[warmup]
            probe_acc  = linear_probe_accuracy(H_t, data.y,
                                               data.train_mask, data.test_mask)
            cka        = representation_similarity(H_t, H_final)

            # TGS with retirement starting at this warmup
            set_seed(seed)
            tg  = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
            est = GradientNormEstimator(m0, DEVICE,
                    edge_index=data.edge_index, num_nodes=data.num_nodes,
                    alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
            model_tgs = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
            opt_tgs   = torch.optim.Adam(
                            list(model_tgs.parameters()) + [est.edge_weights],
                            lr=0.01, weight_decay=5e-4)
            sched = AdaptiveRetirementScheduler(tg,
                        epsilon_max=5e-3, epsilon_min=1e-5, anneal_steps=100,
                        warmup_steps=warmup, max_retire_frac=0.10,
                        max_sparsity=0.65, retire_every=2)
            bv = bt = 0.0
            for epoch in range(EPOCHS):
                model_tgs.train(); am = tg.active_mask
                F.cross_entropy(
                    model_tgs(data.x, tg.edge_index, est.edge_weights[am])[data.train_mask],
                    data.y[data.train_mask]).backward()
                est.update_influence(am); opt_tgs.step(); opt_tgs.zero_grad()
                model_tgs.eval()
                with torch.no_grad(): out = model_tgs(data.x, tg.edge_index)
                p  = out.argmax(-1)
                va = (p[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
                ta = (p[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                sched.update_val_acc(va); sched.step(est.influence_scores(am)); tg.step()
                if va > bv: bv, bt = va, ta
            tgs_acc = bt; sp = tg.sparsity

            # Static at matched sparsity
            n_rem = int(m0 * sp)
            _, sidx = score.sort(descending=True)
            rm  = set(sidx[:n_rem].tolist())
            ei_s = data.edge_index[:, torch.tensor([i not in rm for i in range(m0)], dtype=torch.bool)]
            set_seed(seed)
            ms  = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
            os_ = torch.optim.Adam(ms.parameters(), lr=0.01, weight_decay=5e-4)
            bvs = bts = 0.0
            for e in range(EPOCHS):
                ms.train()
                F.cross_entropy(ms(data.x, ei_s)[data.train_mask],
                                data.y[data.train_mask]).backward()
                os_.step(); os_.zero_grad()
                ms.eval()
                with torch.no_grad(): out = ms(data.x, ei_s)
                p  = out.argmax(-1)
                va = (p[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
                ta = (p[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                if va > bvs: bvs, bts = va, ta
            static_acc = bts

            delta = tgs_acc - static_acc
            print(f"    {warmup:>7} {probe_acc:>7.4f} {cka:>6.3f} {tgs_acc:>7.4f} {static_acc:>8.4f} {delta:>+9.4f}")

            seed_results.append({
                "warmup": warmup, "seed": seed,
                "probe_acc": float(probe_acc), "cka": float(cka),
                "tgs_acc": float(tgs_acc), "static_acc": float(static_acc),
                "delta": float(delta), "sparsity": float(sp),
                "dense_acc": float(dense_acc),
            })

        # Compute correlation: probe_acc -> delta
        probes = [r["probe_acc"] for r in seed_results]
        deltas = [r["delta"]     for r in seed_results]
        r_p, p_p = pearsonr(probes, deltas)
        r_s, p_s = spearmanr(probes, deltas)
        print(f"\n    Correlation probe_acc -> TGS_gain:")
        print(f"      Pearson  r={r_p:+.4f}  p={p_p:.4f}")
        print(f"      Spearman r={r_s:+.4f}  p={p_s:.4f}")

        all_results.extend(seed_results)

    # Aggregate across seeds
    print(f"\n  Aggregated (mean ± std across {len(SEEDS)} seeds):")
    print(f"  {'Warmup':>7} {'Probe':>9} {'TGS':>9} {'Static':>9} {'Delta':>9}")
    by_warmup = {}
    for r in all_results:
        w = r["warmup"]
        by_warmup.setdefault(w, []).append(r)

    agg_rows = []
    for w in WARMUP_GRID:
        rs = by_warmup[w]
        probe_m = np.mean([r["probe_acc"] for r in rs])
        probe_s = np.std([r["probe_acc"]  for r in rs])
        tgs_m   = np.mean([r["tgs_acc"]   for r in rs])
        tgs_s   = np.std([r["tgs_acc"]    for r in rs])
        stat_m  = np.mean([r["static_acc"] for r in rs])
        delta_m = np.mean([r["delta"]      for r in rs])
        delta_s = np.std([r["delta"]       for r in rs])
        print(f"  {w:>7} {probe_m:>7.4f}±{probe_s:.3f} {tgs_m:>7.4f}±{tgs_s:.3f} "
              f"{stat_m:>7.4f}  {delta_m:>+7.4f}±{delta_s:.3f}")
        agg_rows.append({
            "warmup": w, "probe_mean": float(probe_m), "probe_std": float(probe_s),
            "tgs_mean": float(tgs_m), "tgs_std": float(tgs_s),
            "static_mean": float(stat_m), "delta_mean": float(delta_m), "delta_std": float(delta_s),
        })

    # Overall correlation
    all_probes = [r["probe_acc"] for r in all_results]
    all_deltas = [r["delta"]     for r in all_results]
    r_all, p_all = pearsonr(all_probes, all_deltas)
    print(f"\n  Overall correlation (all seeds+warmups): Pearson r={r_all:+.4f} p={p_all:.4f}")

    return {"dataset": ds_name, "raw": all_results, "aggregated": agg_rows,
            "overall_pearson": float(r_all), "overall_p": float(p_all)}


def main():
    all_ds_results = {}
    for ds in ["Cora", "CiteSeer", "PubMed"]:
        result = run_dataset(ds)
        all_ds_results[ds] = result

    # Cross-dataset: does probe gain predict TGS advantage?
    print("\n" + "="*65)
    print("Cross-dataset: Probe Gain -> TGS Gain")
    print("="*65)
    print(f"{'Dataset':<12} {'PearsonR':>10} {'p-val':>8} {'MaxProbe@40':>12} {'MaxDelta':>10}")
    for ds, res in all_ds_results.items():
        agg = res["aggregated"]
        row40 = next((r for r in agg if r["warmup"]==40), agg[-1])
        max_delta = max(r["delta_mean"] for r in agg)
        print(f"{ds:<12} {res['overall_pearson']:>+10.4f} {res['overall_p']:>8.4f} "
              f"{row40['probe_mean']:>12.4f} {max_delta:>+10.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/representation_information.json", "w") as f:
        json.dump(all_ds_results, f, indent=2, default=float)
    print("\nSaved results/representation_information.json")


if __name__ == "__main__":
    main()
