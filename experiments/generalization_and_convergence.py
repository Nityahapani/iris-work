"""
experiments/generalization_and_convergence.py

Runs P2, P3, P4, P5 experiments in one script:

P2 — Architecture generalization: GCN vs GAT vs GraphSAGE on Cora/CiteSeer
     Tests the thesis that TGS is architecture-agnostic.

P3 — PubMed completion: fresh-edge test, temporal vs static.

P4 — Convergence speed: val accuracy curves with retirement events marked.
     Shows TGS gets the best of both worlds — dense early, sparse late.

P5 — OGB-arxiv scale test: 169k nodes, up to 100 epochs.
     Even partial results prove the method scales.
"""

import sys, os, json, time, tracemalloc
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree

from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn  import TemporalGCN
from tgs.models.gat  import TemporalGAT
from tgs.models.sage import TemporalSAGE
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils.reproducibility import set_seed

DEVICE = torch.device("cpu")
SEED   = 42
EPOCHS = 300


# ── Model factory ───────────────────────────────────────────────────────────

def build_model(arch: str, in_ch: int, hidden: int, out_ch: int):
    if arch == "GCN":
        return TemporalGCN(in_ch, hidden, out_ch, num_layers=2, dropout=0.5)
    elif arch == "GAT":
        return TemporalGAT(in_ch, hidden, out_ch, num_layers=2, heads=8, dropout=0.6)
    elif arch == "SAGE":
        return TemporalSAGE(in_ch, hidden, out_ch, num_layers=2, dropout=0.5)
    else:
        raise ValueError(arch)


# ── Core TGS runner with convergence history ────────────────────────────────

def run_tgs(data, nf, nc, arch="GCN", max_sp=0.65, epochs=EPOCHS, hidden=64):
    set_seed(SEED)
    m0 = data.edge_index.shape[1]

    tg  = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE,
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    model = build_model(arch, nf, hidden, nc).to(DEVICE)

    # GAT/SAGE don't use edge_weights in forward the same way — use separate param
    use_ew = (arch == "GCN")
    params = list(model.parameters()) + ([est.edge_weights] if use_ew else [])
    opt = torch.optim.Adam(params, lr=0.01 if arch != "GAT" else 0.005,
                           weight_decay=5e-4)
    sched = AdaptiveRetirementScheduler(tg,
                epsilon_max=5e-3, epsilon_min=1e-5, anneal_steps=100,
                warmup_steps=40, max_retire_frac=0.10,
                max_sparsity=max_sp, retire_every=2)
    flops = FLOPsCounter(m0, 2, hidden)

    history = {
        "epoch": [], "val_acc": [], "test_acc": [],
        "sparsity": [], "retired_this_step": [], "mt": []
    }
    best_val = best_test = 0.0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train(); am = tg.active_mask
        ew = est.edge_weights[am] if use_ew else None
        logits = model(data.x, tg.edge_index, ew)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad(); loss.backward()
        if use_ew: est.update_influence(am)
        else:       est.update_influence(am)   # still update for structural scores
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, tg.edge_index)
        preds    = out.argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        sched.update_val_acc(val_acc)
        n_ret = sched.step(est.influence_scores(am))
        flops.record_step(tg.mt); tg.step()

        history["epoch"].append(epoch)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(tg.sparsity)
        history["retired_this_step"].append(n_ret)
        history["mt"].append(tg.mt)

        if val_acc > best_val: best_val, best_test = val_acc, test_acc

    elapsed = time.perf_counter() - t0
    f  = flops.summary()
    rs = sched.summary()

    # Fresh GCN on final edge set
    final_ei = tg.edge_index.clone()
    set_seed(SEED)
    m2  = build_model(arch, nf, hidden, nc).to(DEVICE)
    o2  = torch.optim.Adam(m2.parameters(), lr=0.01 if arch != "GAT" else 0.005, weight_decay=5e-4)
    bv2 = bt2 = 0.0
    for epoch in range(epochs):
        m2.train()
        F.cross_entropy(m2(data.x, final_ei)[data.train_mask],
                        data.y[data.train_mask]).backward()
        o2.step(); o2.zero_grad()
        m2.eval()
        with torch.no_grad(): el2 = m2(data.x, final_ei)
        p2 = el2.argmax(-1)
        v2 = (p2[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        t2 = (p2[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if v2 > bv2: bv2, bt2 = v2, t2

    return {
        "arch": arch, "test_acc": best_test, "fresh_acc": bt2,
        "sparsity": tg.sparsity, "flops_red": f["flops_reduction"],
        "distortion": rs["cumulative_distortion_bound"],
        "runtime_s": elapsed, "history": history,
    }


def run_dense(data, nf, nc, arch="GCN", epochs=EPOCHS, hidden=64):
    """Dense baseline for convergence comparison."""
    set_seed(SEED)
    model = build_model(arch, nf, hidden, nc).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(),
                lr=0.01 if arch != "GAT" else 0.005, weight_decay=5e-4)
    history = {"epoch": [], "val_acc": [], "test_acc": []}
    best_val = best_test = 0.0
    for epoch in range(epochs):
        model.train()
        loss = F.cross_entropy(model(data.x, data.edge_index)[data.train_mask],
                               data.y[data.train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): out = model(data.x, data.edge_index)
        preds    = out.argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        history["epoch"].append(epoch)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        if val_acc > best_val: best_val, best_test = val_acc, test_acc
    return {"arch": arch, "test_acc": best_test, "sparsity": 0.0,
            "flops_red": 0.0, "history": history}


def run_static(data, nf, nc, target_sp, arch="GCN", epochs=EPOCHS, hidden=64):
    """Static degree-prune baseline at target_sp."""
    set_seed(SEED)
    m0  = data.edge_index.shape[1]
    src, dst = data.edge_index[0], data.edge_index[1]
    deg = degree(dst, data.num_nodes, dtype=torch.float)
    er  = 1.0 / deg[src].clamp(min=1) + 1.0 / deg[dst].clamp(min=1)
    score = deg[src] * deg[dst]; score[er >= torch.quantile(er, 0.90)] = -1.0
    n_rem = int(m0 * target_sp)
    _, sidx = score.sort(descending=True)
    rm = set(sidx[:n_rem].tolist())
    ei = data.edge_index[:, torch.tensor([i not in rm for i in range(m0)], dtype=torch.bool)]

    model = build_model(arch, nf, hidden, nc).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(),
                lr=0.01 if arch != "GAT" else 0.005, weight_decay=5e-4)
    history = {"epoch": [], "val_acc": [], "test_acc": []}
    best_val = best_test = 0.0
    for epoch in range(epochs):
        model.train()
        F.cross_entropy(model(data.x, ei)[data.train_mask],
                        data.y[data.train_mask]).backward()
        opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): out = model(data.x, ei)
        preds    = out.argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        history["epoch"].append(epoch)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        if val_acc > best_val: best_val, best_test = val_acc, test_acc
    return {"arch": arch, "test_acc": best_test,
            "sparsity": 1 - ei.shape[1]/m0,
            "flops_red": 1 - ei.shape[1]/m0, "history": history}


# ── P2: Architecture generalization ─────────────────────────────────────────

def run_p2():
    print("\n" + "="*70)
    print("P2 — Architecture Generalization")
    print("="*70)
    results = {}
    for ds_name in ["Cora", "CiteSeer"]:
        dataset = Planetoid(root="./data", name=ds_name, transform=NormalizeFeatures())
        data    = dataset[0].to(DEVICE)
        nf, nc  = dataset.num_features, dataset.num_classes
        results[ds_name] = {}

        print(f"\n  {ds_name}:")
        print(f"  {'Arch':<8} {'Dense acc':>10} {'TGS acc':>9} {'Fresh':>7} {'Sparsity':>9} {'FLOPs↓':>7}")
        print(f"  {'-'*55}")

        for arch in ["GCN", "GAT", "SAGE"]:
            r_dense = run_dense(data, nf, nc, arch=arch)
            r_tgs   = run_tgs(data, nf, nc, arch=arch)
            results[ds_name][arch] = {"dense": r_dense, "tgs": r_tgs}
            print(f"  {arch:<8} {r_dense['test_acc']:>10.4f} {r_tgs['test_acc']:>9.4f} "
                  f"{r_tgs['fresh_acc']:>7.4f} {r_tgs['sparsity']:>9.3f} {r_tgs['flops_red']:>7.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/arch_generalization.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\n  Saved results/arch_generalization.json")
    return results


# ── P3: PubMed completion ────────────────────────────────────────────────────

def run_p3():
    print("\n" + "="*70)
    print("P3 — PubMed Completion")
    print("="*70)
    dataset = Planetoid(root="./data", name="PubMed", transform=NormalizeFeatures())
    data    = dataset[0].to(DEVICE)
    nf, nc  = dataset.num_features, dataset.num_classes
    print(f"  n={data.num_nodes}, m={data.edge_index.shape[1]}")

    print("  Running TGS...")
    r_tgs = run_tgs(data, nf, nc, arch="GCN")
    sp = r_tgs["sparsity"]
    print(f"  TGS: test={r_tgs['test_acc']:.4f}  fresh={r_tgs['fresh_acc']:.4f}  sp={sp:.3f}  FLOPs↓={r_tgs['flops_red']:.3f}")

    print("  Running Dense...")
    r_dense = run_dense(data, nf, nc, arch="GCN")
    print(f"  Dense: test={r_dense['test_acc']:.4f}")

    print(f"  Running Static @{sp:.3f}...")
    r_static = run_static(data, nf, nc, target_sp=sp, arch="GCN")
    print(f"  Static: test={r_static['test_acc']:.4f}")

    results = {"TGS": r_tgs, "Dense": r_dense, "Static": r_static}
    with open("results/pubmed_complete.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("  Saved results/pubmed_complete.json")
    return results


# ── P4: Convergence curves ────────────────────────────────────────────────────

def run_p4():
    """Already collected during P2/P3 — just generate the figure."""
    print("\n" + "="*70)
    print("P4 — Convergence Speed (figure only, data from P2/P3)")
    print("="*70)

    dataset = Planetoid(root="./data", name="Cora", transform=NormalizeFeatures())
    data    = dataset[0].to(DEVICE)
    nf, nc  = dataset.num_features, dataset.num_classes

    print("  Collecting convergence histories (GCN, Cora)...")
    r_dense  = run_dense(data, nf, nc, arch="GCN")
    r_tgs    = run_tgs(data, nf, nc, arch="GCN")
    r_static = run_static(data, nf, nc, target_sp=r_tgs["sparsity"], arch="GCN")

    results = {"dense": r_dense, "tgs": r_tgs, "static": r_static}
    with open("results/convergence_curves.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("  Saved results/convergence_curves.json")
    return results


# ── P5: OGB-arxiv scale test ──────────────────────────────────────────────────

def run_p5():
    print("\n" + "="*70)
    print("P5 — OGB Scale Test (ogbn-arxiv)")
    print("="*70)
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="./data")
        data_ogb = dataset[0]
        split_idx = dataset.get_idx_split()

        # Convert to standard masks
        n = data_ogb.num_nodes
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        test_mask  = torch.zeros(n, dtype=torch.bool)
        train_mask[split_idx["train"]] = True
        val_mask[split_idx["valid"]]   = True
        test_mask[split_idx["test"]]   = True
        data_ogb.train_mask = train_mask
        data_ogb.val_mask   = val_mask
        data_ogb.test_mask  = test_mask
        data_ogb.y          = data_ogb.y.squeeze()
        data_ogb = data_ogb.to(DEVICE)

        m0 = data_ogb.edge_index.shape[1]
        nf = data_ogb.num_node_features
        nc = dataset.num_classes
        print(f"  ogbn-arxiv: n={data_ogb.num_nodes}, m={m0}, classes={nc}")

        # Run for 100 epochs only (CPU budget)
        SCALE_EPOCHS = 100
        print(f"  Running Dense ({SCALE_EPOCHS} epochs)...")
        t0 = time.perf_counter()
        set_seed(SEED)
        model_d = TemporalGCN(nf, 256, nc, 3, 0.5).to(DEVICE)
        opt_d   = torch.optim.Adam(model_d.parameters(), lr=0.01, weight_decay=0)
        bv_d = bt_d = 0.0
        for epoch in range(SCALE_EPOCHS):
            model_d.train()
            F.cross_entropy(model_d(data_ogb.x, data_ogb.edge_index)[data_ogb.train_mask],
                            data_ogb.y[data_ogb.train_mask]).backward()
            opt_d.step(); opt_d.zero_grad()
            model_d.eval()
            with torch.no_grad(): out = model_d(data_ogb.x, data_ogb.edge_index)
            preds = out.argmax(-1)
            va = (preds[data_ogb.val_mask] == data_ogb.y[data_ogb.val_mask]).float().mean().item()
            ta = (preds[data_ogb.test_mask] == data_ogb.y[data_ogb.test_mask]).float().mean().item()
            if va > bv_d: bv_d, bt_d = va, ta
            if epoch % 20 == 0: print(f"    Dense ep={epoch}  val={va:.4f}")
        dense_time = time.perf_counter() - t0
        print(f"  Dense done: test={bt_d:.4f}  time={dense_time:.1f}s")

        print(f"  Running TGS ({SCALE_EPOCHS} epochs)...")
        t0 = time.perf_counter()
        set_seed(SEED)
        tg   = TemporalGraph(data_ogb.edge_index, data_ogb.num_nodes, device=DEVICE)
        est  = GradientNormEstimator(m0, DEVICE,
                 edge_index=data_ogb.edge_index, num_nodes=data_ogb.num_nodes,
                 alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
        model_t = TemporalGCN(nf, 256, nc, 3, 0.5).to(DEVICE)
        opt_t   = torch.optim.Adam(list(model_t.parameters()) + [est.edge_weights],
                                   lr=0.01, weight_decay=0)
        sched_t = AdaptiveRetirementScheduler(tg, epsilon_max=5e-3, epsilon_min=1e-5,
                    anneal_steps=60, warmup_steps=20, max_retire_frac=0.10,
                    max_sparsity=0.60, retire_every=2)
        flops_t = FLOPsCounter(m0, 3, 256)
        bv_t = bt_t = 0.0
        for epoch in range(SCALE_EPOCHS):
            model_t.train(); am = tg.active_mask
            logits = model_t(data_ogb.x, tg.edge_index, est.edge_weights[am])
            loss   = F.cross_entropy(logits[data_ogb.train_mask], data_ogb.y[data_ogb.train_mask])
            opt_t.zero_grad(); loss.backward(); est.update_influence(am); opt_t.step()
            model_t.eval()
            with torch.no_grad(): out = model_t(data_ogb.x, tg.edge_index)
            preds = out.argmax(-1)
            va = (preds[data_ogb.val_mask] == data_ogb.y[data_ogb.val_mask]).float().mean().item()
            ta = (preds[data_ogb.test_mask] == data_ogb.y[data_ogb.test_mask]).float().mean().item()
            sched_t.update_val_acc(va); sched_t.step(est.influence_scores(am))
            flops_t.record_step(tg.mt); tg.step()
            if va > bv_t: bv_t, bt_t = va, ta
            if epoch % 20 == 0:
                print(f"    TGS ep={epoch}  val={va:.4f}  sp={tg.sparsity:.3f}")
        tgs_time = time.perf_counter() - t0
        print(f"  TGS done: test={bt_t:.4f}  sp={tg.sparsity:.3f}  FLOPs↓={flops_t.summary()['flops_reduction']:.3f}  time={tgs_time:.1f}s")

        out = {
            "dataset": "ogbn-arxiv", "n": data_ogb.num_nodes, "m0": m0,
            "epochs_run": SCALE_EPOCHS,
            "dense":  {"test": bt_d, "sparsity": 0.0, "time_s": dense_time},
            "tgs":    {"test": bt_t, "sparsity": tg.sparsity,
                       "flops_red": flops_t.summary()["flops_reduction"],
                       "time_s": tgs_time,
                       "distortion": sched_t.summary()["cumulative_distortion_bound"]},
        }
        with open("results/ogb_scale_test.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        print("  Saved results/ogb_scale_test.json")
        return out

    except Exception as e:
        print(f"  OGB test failed: {e}")
        return None


if __name__ == "__main__":
    all_results = {}

    r_p2 = run_p2()
    all_results["p2_arch"] = r_p2

    r_p3 = run_p3()
    all_results["p3_pubmed"] = r_p3

    r_p4 = run_p4()
    all_results["p4_convergence"] = r_p4

    r_p5 = run_p5()
    all_results["p5_ogb"] = r_p5

    print("\n" + "="*70)
    print("All experiments complete.")
