"""Shared factory module for train.py and evaluate.py.

Single source of truth for:
  - Config loading (thin wrapper over openood.utils.setup_config)
  - Benchmark construction (re-exported from utils.factory.get_benchmark)
  - Model construction (re-exported from utils.factory.init_model)
  - Optimizer construction (build_optimizer dispatch)

Both pipelines import exclusively from this module to guarantee identical
initialisation semantics.
"""
from __future__ import annotations

from torch.optim import SGD, Adam, AdamW, Optimizer

from openood.utils import setup_config as _openood_setup_config
from utils.factory import get_benchmark as _get_benchmark
from utils.factory import init_model as _init_model

__all__ = [
    "setup_config",
    "build_optimizer",
    "init_model",
    "get_benchmark",
]


OPTIMIZER_MAP = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
}


def setup_config():
    """Thin re-export of openood.utils.setup_config so both pipelines have a
    single import point."""
    return _openood_setup_config()


def get_benchmark(config):
    """Re-export of utils.factory.get_benchmark."""
    return _get_benchmark(config)


def init_model(config, benchmark):
    """Re-export of utils.factory.init_model."""
    return _init_model(config, benchmark)


def build_optimizer(config, model) -> Optimizer:
    """Construct a torch optimizer from ``config.optimizer``.

    Supported names (case-insensitive): ``adam``, ``adamw``, ``sgd``.
    Raises ``ValueError`` for any other name.
    """
    name = config.optimizer.name.lower()
    try:
        cls = OPTIMIZER_MAP[name]
    except KeyError:
        raise ValueError(f"Invalid optimizer: {config.optimizer.name}") from None

    if cls is SGD:
        return cls(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=config.optimizer.nesterov,
        )
    return cls(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
