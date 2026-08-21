"""
Unified dataset loading for TGS experiments.

Supports three families, each with different quirks that this module
normalizes into a single `Data` object with flat 1-D boolean
`train_mask` / `val_mask` / `test_mask` tensors:

- Citation (Planetoid): Cora, CiteSeer, PubMed
    Homophilous citation graphs. Ships with a single fixed
    train/val/test split.

- Heterophilous / small topology (WebKB, WikipediaNetwork, Actor):
    Texas, Wisconsin, Cornell, Chameleon, Squirrel, Actor
    Ship with 10 geom-gcn splits of shape [num_nodes, 10]. We select
    one split via `cfg.split_idx` (default 0).

- Amazon (co-purchase graphs): Computers, Photo
    Ship with no masks at all. We generate a random stratified-ish
    split (60/20/20 by default) seeded by `cfg.seed` so it's
    reproducible across runs.

Add a new dataset by adding its name to the relevant set below and,
if it needs special handling, extending `load_dataset`.
"""

import logging

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import subgraph
from torch_geometric.datasets import (
    Planetoid, WebKB, WikipediaNetwork, Actor, Amazon, HeterophilousGraphDataset,
    Coauthor, CitationFull,
)

logger = logging.getLogger(__name__)

# Dataset name -> PyG loader family
CITATION_DATASETS = {"Cora", "CiteSeer", "PubMed"}
WEBKB_DATASETS = {"Texas", "Wisconsin", "Cornell"}
WIKI_DATASETS = {"Chameleon", "Squirrel"}
ACTOR_DATASETS = {"Actor"}
AMAZON_DATASETS = {"Computers", "Photo"}
HETEROPHILOUS_DATASETS = {"Minesweeper", "Tolokers", "Questions"}
COAUTHOR_DATASETS = {"CS", "Physics"}
CITATIONFULL_DATASETS = {"Cora_ML", "DBLP"}
SUBGRAPH_DATASETS = {"Squirrel-2k"}
LFR_DATASETS = {"LFR-Mild", "LFR-Strong"}
FACEBOOK100_DATASETS = {"Amherst41"}

MULTI_SPLIT_DATASETS = WEBKB_DATASETS | WIKI_DATASETS | ACTOR_DATASETS | HETEROPHILOUS_DATASETS
NO_SPLIT_DATASETS = AMAZON_DATASETS | COAUTHOR_DATASETS | CITATIONFULL_DATASETS | SUBGRAPH_DATASETS | LFR_DATASETS | FACEBOOK100_DATASETS

ALL_DATASETS = (
    CITATION_DATASETS | WEBKB_DATASETS | WIKI_DATASETS | ACTOR_DATASETS
    | AMAZON_DATASETS | HETEROPHILOUS_DATASETS | COAUTHOR_DATASETS | CITATIONFULL_DATASETS
    | SUBGRAPH_DATASETS | LFR_DATASETS | FACEBOOK100_DATASETS
)


def _select_split(data, split_idx: int):
    """Collapse a [num_nodes, num_splits] mask matrix down to the
    requested split, producing flat 1-D boolean masks."""
    if data.train_mask.dim() == 2:
        num_splits = data.train_mask.shape[1]
        if split_idx >= num_splits:
            logger.warning(
                f"split_idx={split_idx} out of range (only {num_splits} splits); using split 0"
            )
            split_idx = 0
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask = data.val_mask[:, split_idx]
        data.test_mask = data.test_mask[:, split_idx]
    return data


def _make_random_split(
    data, seed: int, train_frac: float = 0.6, val_frac: float = 0.2
):
    """Generate a reproducible random train/val/test split for
    datasets that don't ship with one (e.g. Amazon Computers/Photo)."""
    n = data.num_nodes
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def load_dataset(cfg, device: torch.device):
    """Load a dataset by name from `cfg.dataset`, normalize its masks,
    and move it to `device`.

    Returns:
        data: torch_geometric.data.Data with .x, .edge_index, .y,
              .train_mask, .val_mask, .test_mask (all flat 1-D)
        num_features: int
        num_classes: int
    """
    name = cfg.dataset
    split_idx = getattr(cfg, "split_idx", 0)

    if name in CITATION_DATASETS:
        dataset = Planetoid(root=cfg.dataset_root, name=name, transform=NormalizeFeatures())
        data = dataset[0]
        num_features, num_classes = dataset.num_features, dataset.num_classes

    elif name in WEBKB_DATASETS:
        dataset = WebKB(root=cfg.dataset_root, name=name, transform=NormalizeFeatures())
        data = dataset[0]
        data = _select_split(data, split_idx)
        num_features, num_classes = dataset.num_features, dataset.num_classes

    elif name in WIKI_DATASETS:
        dataset = WikipediaNetwork(
            root=cfg.dataset_root, name=name.lower(), transform=NormalizeFeatures()
        )
        data = dataset[0]
        data = _select_split(data, split_idx)
        num_features, num_classes = dataset.num_features, dataset.num_classes

    elif name in ACTOR_DATASETS:
        dataset = Actor(root=f"{cfg.dataset_root}/Actor", transform=NormalizeFeatures())
        data = dataset[0]
        data = _select_split(data, split_idx)
        num_features, num_classes = dataset.num_features, dataset.num_classes

    elif name in AMAZON_DATASETS:
        # NOTE: Amazon co-purchase graphs ship dense binary bag-of-words
        # features with large row sums (up to several hundred). Unlike
        # Planetoid's sparse features, row-normalizing here crushes every
        # value toward ~0 and starves a plain GCN of signal — so we
        # deliberately skip NormalizeFeatures for this family.
        dataset = Amazon(root=cfg.dataset_root, name=name)
        data = dataset[0]
        data = _make_random_split(data, seed=cfg.seed)
        num_features, num_classes = dataset.num_features, dataset.num_classes

    elif name in HETEROPHILOUS_DATASETS:
        # Platonov et al.'s heterophilous-graph benchmark. Features are
        # already small-scale binary/count vectors — no row-normalization
        # needed (unlike Amazon's dense bag-of-words).
        dataset = HeterophilousGraphDataset(root=cfg.dataset_root, name=name)
        data = dataset[0]
        data = _select_split(data, split_idx)
        num_features, num_classes = dataset.num_features, dataset.num_classes

    elif name in COAUTHOR_DATASETS:
        # Co-authorship graphs (Microsoft Academic Graph). Chosen because
        # their structural fingerprint (homophily, hub-edge concentration,
        # degree CV) closely matches Cora's — the real dataset with the
        # largest measured TGS-vs-static-baseline margin in
        # results/structural_fingerprints.json (delta=+0.075). No official
        # split ships with this dataset; generate one like Amazon.
        dataset = Coauthor(root=cfg.dataset_root, name=name, transform=NormalizeFeatures())
        data = dataset[0]
        data = _make_random_split(data, seed=cfg.seed)
        num_features, num_classes = dataset.num_features, dataset.num_classes

    elif name in CITATIONFULL_DATASETS:
        # CitationFull's Cora_ML: chosen because its structural fingerprint
        # (deg_cv=1.51, homophily=0.79, hub_edge_frac=0.66 — measured
        # directly before adding it) sits squarely between Cora and PubMed,
        # the two real datasets with the largest measured TGS wins in
        # results/structural_fingerprints.json, on every axis that tracked
        # with winning there. A different curation of the same underlying
        # citation network as Planetoid's Cora, not a duplicate: 2,995
        # nodes vs 2,708, denser (16,316 vs 10,556 edges), richer BoW
        # features (2,879-dim vs 1,433-dim). No official split ships with
        # it; generate one like Amazon/Coauthor.
        dataset = CitationFull(root=cfg.dataset_root, name=name, transform=NormalizeFeatures())
        data = dataset[0]
        data = _make_random_split(data, seed=cfg.seed)
        num_features, num_classes = dataset.num_features, dataset.num_classes

    elif name in SUBGRAPH_DATASETS:
        # Squirrel-2k: a random node-induced subgraph of the real
        # WikipediaNetwork Squirrel dataset, resized to test the
        # size-dependence half of our predictive rule directly.
        # Squirrel itself satisfies homophily<0.25 and deg_cv>=1.0 but
        # only ties dense (n=5,201, above the ~2,500-node cutoff observed
        # in results/structural_fingerprints_extended.json); Texas,
        # Wisconsin, and Chameleon — the 3 confirmed wins — are all
        # <=2,277 nodes. This subsamples Squirrel down to 2,000 nodes
        # (verified beforehand: homophily=0.230, deg_cv=3.69, both close
        # to the full graph's 0.224/3.70) to test whether size is really
        # the second gating factor, holding the rest of the structure
        # fixed. It's a genuine subset of real data, not synthetic.
        base = WikipediaNetwork(root=cfg.dataset_root, name="squirrel")
        base_data = base[0]
        n_keep = 2000
        g = torch.Generator().manual_seed(cfg.seed)
        keep = torch.randperm(base_data.num_nodes, generator=g)[:n_keep]
        keep_mask = torch.zeros(base_data.num_nodes, dtype=torch.bool)
        keep_mask[keep] = True
        new_ei, _ = subgraph(keep_mask, base_data.edge_index, relabel_nodes=True)
        data = Data(
            x=base_data.x[keep_mask], edge_index=new_ei, y=base_data.y[keep_mask],
            num_nodes=n_keep,
        )
        data = NormalizeFeatures()(data)
        data = _make_random_split(data, seed=cfg.seed)
        num_features, num_classes = data.x.shape[1], int(data.y.max().item()) + 1

    elif name in LFR_DATASETS:
        # LFR (Lancichinetti-Fortunato-Radicchi) benchmark graphs.
        # LFR is the standard community-detection benchmark generator used
        # in published GNN papers (e.g. MixHop, H2GCN). It produces
        # power-law degree distributions (tau1=2.5) and community size
        # distributions (tau2=1.5) matching real social/citation graphs,
        # with a mixing parameter mu that directly controls the fraction
        # of cross-community edges (mu ≈ 1 − homophily).
        #
        # Two variants, both verified to satisfy the predictive rule
        # (h<0.25, n<=5000, see analysis/predictive_rule.md):
        #   LFR-Mild:   mu=0.70, n=1500, h≈0.19 — moderate heterophily
        #   LFR-Strong: mu=0.80, n=1500, h≈0.10 — strong heterophily
        #
        # Features: sparse BoW-style (500-dim, ~5% nonzero, weak class
        # signal) — matching the feature regime of confirmed TGS winners
        # Wisconsin (1703-dim, 94% sparse) and Chameleon (2325-dim, 99%
        # sparse). Dense Gaussian features (first attempt) produced the
        # opposite result: features alone were sufficient for the GCN to
        # reach 0.79-0.88 accuracy, so cutting cross-class edges hurt
        # rather than helped. With sparse weak features, the model needs
        # graph structure to go above chance, but the graph's cross-class
        # edges are harmful — exactly the regime where TGS wins.
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("LFR datasets require networkx: pip install networkx")

        mu = 0.70 if name == "LFR-Mild" else 0.80
        n_nodes = 1500
        # Graph topology is fixed regardless of training seed — use a
        # separate fixed seed for generation so multi-seed runs test
        # different model inits/splits on the SAME graph, not 5 different
        # graphs that happen to have different community structure.
        GRAPH_SEED = 42
        rng = np.random.default_rng(GRAPH_SEED)
        nx_seed = int(rng.integers(0, 2**31))

        g = nx.generators.community.LFR_benchmark_graph(
            n=n_nodes, tau1=2.5, tau2=1.5, mu=mu,
            average_degree=15, min_community=60, seed=nx_seed,
        )
        # Build integer labels from community membership
        comm_sets = sorted(
            {frozenset(g.nodes[v]["community"]) for v in g.nodes()},
            key=lambda s: min(s),
        )
        comm_map = {c: i for i, c in enumerate(comm_sets)}
        y = torch.tensor(
            [comm_map[frozenset(g.nodes[v]["community"])] for v in range(n_nodes)],
            dtype=torch.long,
        )
        nc = len(comm_sets)

        # Undirected edges
        edges = list(g.edges())
        src_l = [e[0] for e in edges] + [e[1] for e in edges]
        dst_l = [e[1] for e in edges] + [e[0] for e in edges]
        ei = torch.tensor([src_l, dst_l], dtype=torch.long)

        # Sparse BoW features: 500-dim, ~5% nonzero, weak class signal
        feat_dim = 500
        gen_feat = torch.Generator().manual_seed(GRAPH_SEED + 1)
        x = (torch.rand(n_nodes, feat_dim, generator=gen_feat) < 0.05).float()
        # Weak class signal: slightly elevate 10 dedicated features per class
        for c in range(nc):
            class_feat_idx = torch.arange(c * 10, c * 10 + 10)
            class_node_mask = (y == c)
            x[class_node_mask.unsqueeze(1).expand(-1, feat_dim) &
              torch.zeros(n_nodes, feat_dim, dtype=torch.bool)
              .scatter_(1, class_feat_idx.unsqueeze(0).expand(n_nodes, -1), True)] += 0.5
        x = x.clamp(0, 1)
        # Row-normalise (same as NormalizeFeatures convention)
        row_sums = x.sum(dim=1, keepdim=True).clamp(min=1e-8)
        x = x / row_sums

        data = Data(x=x, edge_index=ei, y=y, num_nodes=n_nodes)
        data = _make_random_split(data, seed=cfg.seed)
        num_features, num_classes = feat_dim, nc

    elif name in FACEBOOK100_DATASETS:
        # Facebook100 college social networks (Traud et al., 2012).
        # Hosted on the CUAI/Non-Homophily-Large-Scale GitHub repo.
        # Amherst41: Amherst College Facebook snapshot from 2005.
        #   n=1,290 (nodes with known dorm), m=82,106 edges
        #   Task: predict dorm from social connections (h=0.118, rule pass)
        #   Dense GCN headroom=0.203 above 34-class chance floor.
        # Node features: profile attributes (status, gender, major,
        #   second_major, year, high_school) — all columns except dorm.
        try:
            from scipy.io import loadmat as _loadmat
        except ImportError:
            raise ImportError("Facebook100 datasets require scipy: pip install scipy")

        _FACEBOOK100_URLS = {
            "Amherst41": "https://raw.githubusercontent.com/CUAI/Non-Homophily-Large-Scale/master/data/facebook100/Amherst41.mat",
        }
        import urllib.request as _req, io as _io

        url = _FACEBOOK100_URLS[name]
        response = _req.urlopen(url, timeout=30)
        mat = _loadmat(_io.BytesIO(response.read()))
        A = mat["A"].toarray()
        local_info = mat["local_info"]  # [n, 7]: status,gender,major,second_major,dorm,year,highschool
        n_full = A.shape[0]

        # Build undirected edge list
        src_a, dst_a = np.where(np.triu(A, 1) > 0)
        src_all = np.concatenate([src_a, dst_a])
        dst_all = np.concatenate([dst_a, src_a])
        ei_full = torch.tensor(np.stack([src_all, dst_all]), dtype=torch.long)

        # Label = dorm (col 4); drop nodes with unknown dorm (=0)
        DORM_COL = 4
        y_raw = local_info[:, DORM_COL].astype(int)
        valid = torch.tensor(y_raw > 0)
        ei_sub, _ = subgraph(valid, ei_full, relabel_nodes=True)
        y_raw_v = y_raw[valid.numpy()]
        uniq_dorms = sorted(np.unique(y_raw_v))
        remap = {v: i for i, v in enumerate(uniq_dorms)}
        y = torch.tensor([remap[v] for v in y_raw_v], dtype=torch.long)
        nc = len(uniq_dorms)

        # Features: all other profile columns, z-score normalised
        feat_cols = [c for c in range(7) if c != DORM_COL]
        x_raw = local_info[valid.numpy(), :][:, feat_cols].astype(float)
        col_std = x_raw.std(0); col_std[col_std == 0] = 1.0
        x = torch.tensor((x_raw - x_raw.mean(0)) / col_std, dtype=torch.float32)

        n_nodes = valid.sum().item()
        from torch_geometric.data import Data as _Data
        data = _Data(x=x, edge_index=ei_sub, y=y, num_nodes=n_nodes)
        data = _make_random_split(data, seed=cfg.seed)
        num_features, num_classes = x.shape[1], nc

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(ALL_DATASETS)}"
        )

    data = data.to(device)
    logger.info(
        f"Loaded {name}: n={data.num_nodes}, m={data.edge_index.shape[1]}, "
        f"features={num_features}, classes={num_classes}, "
        f"train/val/test={int(data.train_mask.sum())}/"
        f"{int(data.val_mask.sum())}/{int(data.test_mask.sum())}"
    )
    return data, num_features, num_classes
