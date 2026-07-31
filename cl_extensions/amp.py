"""AMP (mixed-precision) integration for Avalanche supervised strategies.

Replaces the ``types.MethodType`` monkey-patch in the old ``train_cl.py`` with
a proper class-level mixin. ``enable_amp(strategy)`` re-classes the strategy
instance into a dynamically created subclass that inherits from ``AMPMixin``
and the strategy's original template, so ``super().forward()`` works
correctly through the MRO.

Call order per iteration:
    autocast → forward → loss.backward (scaled) → scaler.unscale_
    → after_backward hooks (ClipGradients sees real grads)
    → scaler.step → scaler.update
"""
from __future__ import annotations

import torch


class AMPMixin:
    """Mixin adding AMP semantics to an Avalanche SupervisedTemplate subclass."""

    # Populated by enable_amp() on the instance.
    _amp_scaler: "torch.cuda.amp.GradScaler"

    def forward(self):
        with torch.cuda.amp.autocast():
            result = super().forward()
        # Cast back to float32: Avalanche calls criterion() outside autocast,
        # and losses like BCELoss require input and target to share dtype.
        return result.float()

    def backward(self):
        self._amp_scaler.scale(self.loss).backward(retain_graph=self.retain_graph)
        # Unscale before after_backward hooks so ClipGradients operates on
        # real (not scaled) gradients.
        self._amp_scaler.unscale_(self.optimizer)

    def optimizer_step(self):
        self._amp_scaler.step(self.optimizer)
        self._amp_scaler.update()


def enable_amp(strategy):
    """Re-class ``strategy`` so its type inherits from AMPMixin.

    The dynamically created class inherits from (AMPMixin, original_type),
    so Method Resolution Order makes ``AMPMixin.forward`` call
    ``original_type.forward`` via ``super()``.

    Returns ``strategy`` (mutated in place) for call-site chaining.
    """
    base_cls = type(strategy)
    amp_cls_name = f"AMP{base_cls.__name__}"
    amp_cls = type(amp_cls_name, (AMPMixin, base_cls), {})
    strategy.__class__ = amp_cls
    strategy._amp_scaler = torch.cuda.amp.GradScaler()
    return strategy
