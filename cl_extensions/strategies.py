import torch
import torch.nn.functional as F
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.supervised import DER
from avalanche.training.supervised import ICaRL as ICaRLAVL
from avalanche.training.templates.common_templates import SupervisedTemplate
from avalanche.training.utils import cycle

from utils.dist_utils import unwrap_model
from utils.helpers import make_replay_collate

from .models import TrainEvalModel
from .plugins import ICaRLLossPlugin, NCMClassifier, _ICaRLPlugin


class ICaRL(ICaRLAVL):
    def __init__(
        self,
        *,
        feature_extractor,
        classifier,
        optimizer,
        memory_size: int,
        buffer_transform,
        fixed_memory: bool,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size=None,
        device="cpu",
        plugins=None,
        evaluator=default_evaluator,
        eval_every=-1,
    ):
        model = TrainEvalModel(
            feature_extractor,
            train_classifier=classifier,
            eval_classifier=NCMClassifier(normalize=True),
        )
        criterion = (
            ICaRLLossPlugin()
        )  # iCaRL requires this specific loss (#966)
        icarl = _ICaRLPlugin(
            memory_size,
            buffer_transform,
            fixed_memory,
        )
        if plugins is None:
            plugins = [icarl]
        else:
            plugins += [icarl]
        if isinstance(criterion, SupervisedPlugin):
            plugins += [criterion]
        SupervisedTemplate.__init__(
            self,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )


class MultiTaskDER(DER):
    """
    A DER strategy that is compatible with multi-task scenarios.
    It works by slicing the output of the current model to match the
    shape of the stored logits from previous tasks.
    """

    def __init__(self, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

    def _before_training_exp(self, **kwargs):
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            model = unwrap_model(self.model)

            # Determine the total number of classes (output dimensions) so that
            # the collate function knows how to pad/slice the stored logits.
            if hasattr(model, "train_classifier"):
                # Case 1: TrainEvalModel wrapper
                target_dim = model.train_classifier.out_features
            elif hasattr(model, "classifier"):
                # Case 2: WrapModel / WrapModelMT wrapper.
                # IncrementalClassifier wraps an nn.Linear in .classifier,
                # so out_features may be one level deeper.
                cls = model.classifier
                target_dim = getattr(cls, "out_features", None)
                if target_dim is None:
                    inner = getattr(cls, "classifier", None)
                    target_dim = getattr(inner, "out_features", None)
            else:
                # Fallback: try direct attribute or use the existing logic
                target_dim = getattr(model, "num_classes", None)

            # Final safety check: if we still don't have target_dim, use
            # the dimension of the first logit in the buffer.
            if target_dim is None and len(buffer) > 0:
                target_dim = buffer[0][3].shape[-1]

            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                    num_workers=kwargs.get("num_workers", 0),
                    collate_fn=make_replay_collate(target_dim, pad_value=0.0),
                )
            )
        else:
            self.replay_loader = None

        super(DER, self)._before_training_exp(**kwargs)

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        if self.replay_loader is None:
            return None

        batch_x, batch_y, batch_tid, batch_logits = next(self.replay_loader)
        if self.transform is not None:
            batch_x = self.transform(batch_x)
        batch_x, batch_y, batch_tid, batch_logits = (
            batch_x.to(self.device),
            batch_y.to(self.device),
            batch_tid.to(self.device),
            batch_logits.to(self.device),
        )
        self.mbatch[0] = torch.cat((batch_x, self.mbatch[0]))
        self.mbatch[1] = torch.cat((batch_y, self.mbatch[1]))
        self.mbatch[2] = torch.cat((batch_tid, self.mbatch[2]))
        self.batch_logits = batch_logits
        self.batch_tid = batch_tid

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            if self.replay_loader is not None:
                # DER Loss computation
                self.loss += F.cross_entropy(
                    self.mb_output[self.batch_size_mem :],
                    self.mb_y[self.batch_size_mem :],
                )
                self.loss += self.alpha * F.mse_loss(
                    self.mb_output[
                        : self.batch_size_mem, : self.batch_logits.shape[1]
                    ],
                    self.batch_logits,
                )
                self.loss += self.beta * F.cross_entropy(
                    self.mb_output[: self.batch_size_mem],
                    self.mb_y[: self.batch_size_mem],
                )

                # They are a few difference compared to the autors impl:
                # - Joint forward pass vs. 3 forward passes
                # - One replay batch vs two replay batches
                # - Logits are stored from the non-transformed sample
                #   after training on task vs instantly on transformed sample

            else:
                self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
