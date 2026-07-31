import os

import torch

# ========= CONSTANTS =========
# With torchrun/DDP each worker sets LOCAL_RANK before the script runs,
# so reading it here gives the correct per-process GPU without any
# extra initialisation.  Falls back to cuda:0 (or cpu) for single-GPU runs.


def get_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # Ensure the ordinal is within range
        num_gpus = torch.cuda.device_count()
        if local_rank >= num_gpus:
            # Fallback or error? Usually this means config mismatch.
            # We'll use 0 as a safe fallback if num_gpus > 0
            safe_rank = local_rank % num_gpus if num_gpus > 0 else 0
            return torch.device(f"cuda:{safe_rank}")
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


# Eagerly resolve so that `from utils import device` works at import time.
device = get_device()

OOD_METRIC_List = ["FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT"]
NEAR_OOD = "nearood"
FAR_OOD = "farood"
OOD_TYPES = [NEAR_OOD, FAR_OOD]
