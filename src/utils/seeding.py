from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_random_state(seed: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(seed)
