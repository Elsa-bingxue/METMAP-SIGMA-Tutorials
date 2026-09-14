import os
import random
import numpy as np
import torch


def set_seed(seed=0, *, deterministic=True):
    """Seed all stochastic libraries used by SIGMA and return provenance."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # PyTorch versions before ``warn_only`` was added.
            torch.use_deterministic_algorithms(True)
    return {
        "random_state": seed,
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "deterministic_torch": bool(deterministic),
    }
