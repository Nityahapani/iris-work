"""Config loading from YAML files."""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class Config:
    # Dataset
    dataset: str = "Cora"
    dataset_root: str = "./data"

    # Model
    model: str = "gcn"
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5

    # Training
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 300
    device: str = "cpu"

    # Retirement scheduler
    scheduler: str = "adaptive"          # 'fixed' or 'adaptive'
    epsilon: float = 1e-3                # fixed-ε threshold
    epsilon_max: float = 1e-2            # adaptive: start
    epsilon_min: float = 1e-4            # adaptive: end
    anneal_steps: int = 200              # adaptive: decay steps
    anneal_schedule: str = "cosine"      # 'cosine', 'linear', 'step'
    warmup_steps: int = 50
    max_retire_frac: float = 0.05
    max_sparsity: float = 0.9
    retire_every: int = 5

    # Influence estimator
    influence_method: str = "gradient"   # 'gradient' or 'exact'
    ema_decay: float = 0.9

    # Logging
    log_every: int = 10
    save_dir: str = "./results"
    run_name: str = "tgs_run"

    # Reproducibility
    seed: int = 42


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = Config()
    for k, v in (raw or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
