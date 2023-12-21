from typing import Dict, Any
import random

import wandb
import numpy as np
import torch


def setup_wandb(cfg: Dict[str, Any]) -> None:
    """Initialize the wandb process.

    Args:
        cfg (Dict[str, Any]): run configuration.
    """
    wandb.init(
        project="reward-shaping",
        name=cfg["wandb_name"],
        config=cfg,
        mode=cfg["wandb_mode"],
        id=cfg["wandb_gen_id"],
        resume="never",  # never resume runs for now
    )


def set_seeds(seed: int) -> None:
    """Set experiment seed across libraries.

    Args:
        seed (int): integer seed.
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
