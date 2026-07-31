"""
Avalanche-compatible plugin for Distributed Data Parallel (DDP) training.

Why we need this:
    Avalanche's SupervisedTemplate creates its own DataLoader via
    make_train_dataloader(). Without intervention each DDP worker would receive
    the same mini-batches (same data, just redundant compute). This plugin
    replaces that call with one that injects a DistributedSampler so that each
    process sees a disjoint shard of the training data.

Usage:
    from cl_extensions.ddp_plugin import DDPSamplerPlugin
    ddp_plugin = DDPSamplerPlugin(rank=rank, world_size=world_size)
    strategy.plugins.append(ddp_plugin)

Barrier placement:
    This plugin is appended LAST to strategy.plugins so its hooks fire after
    all other plugins' hooks at each lifecycle point.

    after_training_exp  barrier (inside strategy.train()):
        Ensures all workers finish iCaRL exemplar construction, EWC Fisher
        computation, and any other after_training_exp work before any worker
        is allowed to return from strategy.train().

    before_training_exp barrier (at start of next experience):
        Guards the inter-experience gap: after strategy.train() returns,
        TrainingManager saves the checkpoint + buffer (rank-0) and then
        reloads the buffer on non-rank-0 workers (Fix 3). The barrier here
        ensures no worker begins the next experience's dataloader patching
        before every worker has finished that post-train bookkeeping.
"""

import types
from typing import Optional

import torch.distributed as dist
from avalanche.core import SupervisedPlugin
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _dist_barrier() -> None:
    """Fire a dist.barrier() only when a process group is active."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class DDPSamplerPlugin(SupervisedPlugin):
    """
    Injects a DistributedSampler into Avalanche's training DataLoader and
    provides the synchronisation barriers required for safe CL+DDP training.

    Hooks used:
    - before_training_exp  : barrier (inter-experience sync) + patch
                             make_train_dataloader for this experience.
    - before_training_epoch: calls sampler.set_epoch() for correct shuffle.
    - after_training_exp   : barrier (intra-experience sync — must fire last).
    """

    supports_distributed = True

    def __init__(self, rank: int, world_size: int, num_workers: int = 8):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers
        self._sampler: Optional[DistributedSampler] = None
        self._epoch_counter: int = 0

    # ------------------------------------------------------------------
    # Inter-experience barrier + dataloader patching
    # ------------------------------------------------------------------
    def before_training_exp(self, strategy, num_workers: int = 0, **kwargs):
        # Barrier: ensures every worker has completed the previous experience's
        # post-train work (checkpoint save + buffer reload in TrainingManager)
        # before any worker starts building the new dataloader.  Safe to call
        # on the very first experience (dist.barrier() with all procs present).
        _dist_barrier()

        rank = self.rank
        world_size = self.world_size
        plugin_ref = self
        plugin_ref._epoch_counter = 0  # reset per experience

        nw = plugin_ref.num_workers

        def _make_train_dataloader_ddp(
            self_strat,
            num_workers: int = nw,
            shuffle: bool = True,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            prefetch_factor: int = 4,
            **kw,
        ):
            sampler = DistributedSampler(
                self_strat.adapted_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=True,
            )
            plugin_ref._sampler = sampler

            # AvalancheDataset may expose a collate_fn; preserve it if present.
            collate_fn = getattr(self_strat.adapted_dataset, "collate_fn", None)

            loader_kwargs = dict(
                batch_size=self_strat.train_mb_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(persistent_workers and num_workers > 0),
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )
            if collate_fn is not None:
                loader_kwargs["collate_fn"] = collate_fn

            self_strat.dataloader = DataLoader(
                self_strat.adapted_dataset,
                **loader_kwargs,
            )

        # Bind as an instance method so `self_strat` resolves correctly.
        strategy.make_train_dataloader = types.MethodType(
            _make_train_dataloader_ddp, strategy
        )

    # ------------------------------------------------------------------
    # Advance the sampler epoch so each epoch gets a different shuffle
    # ------------------------------------------------------------------
    def before_training_epoch(self, strategy, **kwargs):
        if self._sampler is not None:
            self._sampler.set_epoch(self._epoch_counter)
            self._epoch_counter += 1

    # ------------------------------------------------------------------
    # Intra-experience sync barrier (fires last among all plugins)
    # ------------------------------------------------------------------
    def after_training_exp(self, strategy, **kwargs):
        # This barrier must fire AFTER all other plugins' after_training_exp
        # (e.g. iCaRL exemplar construction, EWC Fisher computation).
        # That is guaranteed because this plugin is appended last to
        # strategy.plugins in train_cl.py, so Avalanche calls its hooks last.
        #
        # After this barrier, TrainingManager saves the checkpoint + buffer
        # (rank-0 only) and reloads the buffer on non-rank-0 workers.
        # A second barrier inside TrainingManager.train_task guards that save.
        _dist_barrier()
