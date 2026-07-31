import ast
import numbers
import os
import pickle

import avalanche.benchmarks.datasets.imagenet.imagenet as avalanche_imagenet
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from avalanche.benchmarks.classic import (
    SplitCIFAR10,
    SplitCIFAR100,
    SplitImageNet,
    SplitMNIST,
)
from avalanche.benchmarks.utils.flat_data import ConstantSequence
from avalanche.models import (
    IncrementalClassifier,
    MobilenetV1,
    MultiHeadClassifier,
    SimpleMLP,
    initialize_icarl_net,
    make_icarl_net,
)
from avalanche.models.cosine_layer import CosineIncrementalClassifier
from avalanche.models.packnet import PackNetModel
from avalanche.training import BiC
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.supervised import (
    AGEM,
    ER_ACE,
    ER_AML,
    EWC,
    GEM,
    GDumb,
    JointTraining,
    LwF,
    Naive,
    PackNet,
    Replay,
    SynapticIntelligence,
)
from avalanche.training.supervised.der import (
    ClassBalancedBufferWithLogits as ClassBalancedBufferWithLogitsAVL,
)
from avalanche.training.templates.common_templates import SupervisedTemplate
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from cl_extensions import (
    BeRPlugin,
    FOSTERPlugin,
    MRFAPlugin,
    WeightAlignmentPlugin,
)
from cl_extensions.models import TrainEvalModel, WrapModel, WrapModelMT
from cl_extensions.plugins import ClipGradients, NCMClassifier
from cl_extensions.strategies import ICaRL, MultiTaskDER
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.networks.resnet18_224x224 import ResNet18_224x224
from openood.networks.resnet50 import ResNet50
from openood.networks.vit_b_16 import ViT_B_16

# from cl_extensions.models import AdapterMultiHeadClassifier as MultiHeadClassifier
from utils import device
from utils.dist_utils import get_rank, is_dist_available_and_initialized
from utils.helpers import compute_new_ce_loss, forward_rep, set_task_id

# Bypasses Avalanche's strict MD5 hash checking
# 1. Bypass strict MD5 checksums
avalanche_imagenet._verify_archive = lambda root, file, md5: None


# 2. Fix Avalanche's bug by forcing correct default archive filenames
def patched_parse_archives(self):
    # Check for meta.bin in root or meta_root
    meta_path = self.meta_root if self.meta_root else self.root

    if not os.path.isfile(os.path.join(meta_path, "meta.bin")):
        avalanche_imagenet.parse_devkit_archive(self.root)

    if not os.path.isdir(os.path.join(self.root, "train")):
        avalanche_imagenet.parse_train_archive(self.root, file=None)

    if not os.path.isdir(os.path.join(self.root, "val")):
        avalanche_imagenet.parse_val_archive(
            self.root, file=None, meta_root=meta_path
        )


# Apply the patch to AvalancheImageNet
avalanche_imagenet.AvalancheImageNet.parse_archives = patched_parse_archives


def get_milestones(config):
    # Assuming 'config' is your loaded configuration object
    milestones = config.strategy.milestones
    if isinstance(milestones, list) and isinstance(
        milestones[0], numbers.Number
    ):
        return [int(milestone) for milestone in milestones]
    milestones = "".join(milestones)
    if isinstance(milestones, str):
        try:
            # This handles "[30, 60]" -> [30, 60]
            milestones = ast.literal_eval(milestones)
        except (ValueError, SyntaxError):
            # This handles "30,60" -> [30, 60]
            milestones = [int(x) for x in milestones.split(",")]

    # Ensure they are integers (important for the scheduler!)
    config.strategy.milestones = [int(x) for x in milestones]
    return milestones


class LwFCEPenalty(LwF):
    """This wrapper around LwF computes the total loss
    by diminishing the cross-entropy contribution over time,
    as per the paper
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf
    The loss is L_tot = (1/n_exp_so_far) * L_cross_entropy +
                        alpha[current_exp] * L_distillation
    """

    def _before_backward(self, **kwargs):
        # we can update the cross entropy loss here to fix lwf issue
        self.loss = compute_new_ce_loss(self)
        self.loss *= float(1 / (self.clock.train_exp_counter + 1))
        super()._before_backward(**kwargs)

    def _before_training_exp(self, **kwargs):
        super()._before_training_exp(**kwargs)
        for p in self.plugins:
            if hasattr(p, "prev_model") and p.prev_model is not None:
                if hasattr(p.prev_model, "task_id"):
                    p.prev_model.task_id = None


class ClassBalancedBufferWithLogits(ClassBalancedBufferWithLogitsAVL):
    def update(self, strategy: "SupervisedTemplate", **kwargs):
        assert strategy.experience is not None
        set_task_id(strategy.model, strategy.experience.task_label)
        super().update(strategy, **kwargs)


class CosineMultiHeadClassifier(MultiHeadClassifier):
    """A multi-head classifier where each head is a cosine classifier."""

    def __init__(self, in_features, initial_out_features=0):
        super().__init__(in_features, initial_out_features)
        first_head = CosineIncrementalClassifier(
            self.in_features,
            self.starting_out_features,
        )
        self.classifiers["0"] = first_head

    def adaptation(self, experience):
        """If `dataset` contains new tasks, a new head is initialized.

        :param experience: data from the current experience.
        :return:
        """
        super().adaptation(experience)
        task_labels = experience.task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]
        for tid in set(task_labels):
            tid = str(tid)
            if isinstance(self.classifiers[tid], IncrementalClassifier):
                in_features = self.classifiers[tid].in_features
                out_features = self.classifiers[tid].out_features
                self.classifiers[tid] = CosineIncrementalClassifier(
                    in_features, out_features
                )
                self.classifiers[tid].adaptation(experience)


def get_strategy_class(
    config, model, optimizer, criterion, eval_plugin, fc_layer=None
):
    """
    Returns the corresponding strategy class based on the configuration.

    Args:
        config: A configuration object with a 'strategy' attribute containing the strategy name.

    Returns:
        The strategy class or None if the strategy is not found.
    """
    # ExpertGate is not currently supported because it requires a dedicated
    # architecture rather than the shared model interface used below.

    strategy_name = config.strategy.name
    cl_strategy = None
    plugins = None
    n_epochs = config.strategy.train_epochs

    milestone_ratios = [0.45, 0.7, 0.9]
    milestones = [int(n_epochs * r) for r in milestone_ratios]
    lr_gamma = 0.1
    if "milestones" in config.strategy:
        milestones = get_milestones(config)
    if "lr_gamma" in config.strategy:
        lr_gamma = config.strategy.lr_gamma
    scheduler_plugin = LRSchedulerPlugin(
        MultiStepLR(optimizer, milestones=milestones, gamma=lr_gamma)
    )
    print(f"Scheduler Milestones: [{milestones}] (Total Epochs: {n_epochs})")
    if "clip_grad" in config.strategy and config.strategy.clip_grad:
        plugins = [ClipGradients(max_norm=config.strategy.max_norm)]

    if plugins is None:
        plugins = [scheduler_plugin]
    else:
        plugins.append(scheduler_plugin)
    if "ber" in config.strategy and config.strategy.ber.enable:
        ber_cfg = config.strategy.ber
        plugins.append(
            BeRPlugin(
                num_epochs=ber_cfg.num_epochs,
                m_in=ber_cfg.m_in,
                m_out=ber_cfg.m_out,
                beta=ber_cfg.beta,
                lr=float(ber_cfg.lr),
                weight_decay=float(ber_cfg.weight_decay),
                reset_after_eval=ber_cfg.reset_after_eval,
                batch_size=int(config.strategy.train_mb_size),
                val_beta=float(getattr(ber_cfg, "val_beta", None) or 0.002),
                alpha=float(getattr(ber_cfg, "alpha", None) or 0.1),
            )
        )

    buffer_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                config.dataset.image_size, padding=4
            ),  # Standard CIFAR augmentation
            transforms.RandomHorizontalFlip(),
        ]
    )
    # Regularization based strategies
    if strategy_name == "Joint":
        strategy_type = "joint"
        cl_strategy = JointTraining(
            model,
            optimizer,
            criterion,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "Naive":
        cl_strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
        strategy_type = "naive"
    elif strategy_name == "EWC":
        strategy_type = "regularization"
        cl_strategy = EWC(
            model,
            optimizer,
            criterion,
            ewc_lambda=config.strategy.ewc_lambda,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "LwF":
        strategy_type = "regularization"
        cl_strategy = LwFCEPenalty(
            model,
            optimizer,
            criterion,
            alpha=config.strategy.alpha,
            temperature=config.strategy.temperature,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "SynapticIntelligence":
        strategy_type = "regularization"
        cl_strategy = SynapticIntelligence(
            model,
            optimizer,
            criterion,
            si_lambda=config.strategy.si_lambda,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "ER_ACE":
        strategy_type = "Replay"
        cl_strategy = ER_ACE(
            model,
            optimizer,
            criterion,
            mem_size=config.strategy.mem_size,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "ER_AML":
        # Avalanche's ER_AML._before_training_exp builds one replay DataLoader
        # per class with drop_last=True and batch_size=train_mb_size. With a
        # ClassBalancedBuffer the per-class budget shrinks as classes accrue
        # (mem_size / n_classes_seen); once it falls below train_mb_size a
        # per-class loader yields ZERO batches, and avalanche's cycle() helper
        # (`while True: for b in loader: yield b`) then spins forever without
        # yielding -> a CPU busy-loop hang at er_aml.py:149. This strikes at
        # exp 6 for nexp=20 (5 cls/task) and exp 3 for nexp=10 (10 cls/task).
        #
        # Fix: after the original setup runs, rebuild the per-class loaders
        # with drop_last=False and batch_size clamped to the buffer size, so a
        # short buffer still yields one (smaller) batch and cycle() progresses.
        from avalanche.training.supervised import er_aml as _er_aml_module
        from avalanche.training.utils import cycle as _cycle

        if not getattr(
            _er_aml_module.ER_AML._before_training_exp,
            "_drop_last_patched",
            False,
        ):
            _orig_before_training_exp = (
                _er_aml_module.ER_AML._before_training_exp
            )

            def _patched_before_training_exp(self, **kwargs):
                import torch  # bare `torch` is not imported at module scope

                _orig_before_training_exp(self, **kwargs)
                # Only relevant once replay is active (current_experience > 0).
                if (
                    getattr(self, "pos_neg_loaders", None) is not None
                    and self.experience.current_experience > 0
                ):
                    self.pos_neg_loaders = [
                        _cycle(
                            torch.utils.data.DataLoader(
                                group.buffer,
                                batch_size=max(
                                    1,
                                    min(self.train_mb_size, len(group.buffer)),
                                ),
                                shuffle=True,
                                drop_last=False,
                            )
                        )
                        for group in self.storage_policy.buffer_groups.values()
                    ]

            _patched_before_training_exp._drop_last_patched = True
            _er_aml_module.ER_AML._before_training_exp = (
                _patched_before_training_exp
            )

        strategy_type = "Replay"
        cl_strategy = ER_AML(
            model,
            lambda x: forward_rep(model, x),
            optimizer,
            criterion,
            mem_size=config.strategy.mem_size,
            temp=config.strategy.temp,
            base_temp=config.strategy.base_temp,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "DER":
        strategy_type = "Replay"
        cl_strategy = MultiTaskDER(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            mem_size=config.strategy.mem_size,
            alpha=config.strategy.alpha,
            beta=config.strategy.beta,
            batch_size_mem=config.strategy.train_mb_size,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
            transform=buffer_transform,
        )
        cl_strategy.storage_policy = ClassBalancedBufferWithLogits(
            cl_strategy.mem_size, adaptive_size=True
        )
        # cl_strategy.storage_policy = ReservoirSamplingBuffer(
        #     cl_strategy.mem_size
        # )
    elif strategy_name == "AGEM":
        strategy_type = "Replay+Optimization"
        cl_strategy = AGEM(
            model,
            optimizer,
            criterion,
            patterns_per_exp=config.strategy.patterns_per_exp,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "GEM":
        # Upstream Avalanche GEMPlugin calls quadprog.solve_qp on a QP built
        # from the stored task-gradient matrix. With many tasks (CIFAR-100
        # nexp=20) the QP can become degenerate; some quadprog builds return
        # `None` instead of raising, and the next line `np.dot(v, memories_np)`
        # crashes with `TypeError: unsupported operand type(s) for *: 'NoneType'
        # and 'float'`. ExperimentGuard then skips the task and no checkpoint
        # is written, which breaks the downstream evaluate.py.
        # Patch the plugin method to fall back to the raw (unprojected) gradient
        # when the solver fails, so training continues instead of aborting.
        from avalanche.training.plugins import gem as _gem_module

        if not getattr(
            _gem_module.GEMPlugin.solve_quadprog, "_safe_patched", False
        ):
            _orig_solve_quadprog = _gem_module.GEMPlugin.solve_quadprog

            def _safe_solve_quadprog(self, g):
                try:
                    v_star = _orig_solve_quadprog(self, g)
                    if v_star is None:
                        raise ValueError("quadprog returned None")
                    return v_star
                except Exception as exc:
                    print(
                        f"[GEM] solve_quadprog failed ({type(exc).__name__}: "
                        f"{exc}); falling back to raw gradient for this step.",
                        flush=True,
                    )
                    return g.detach().cpu().contiguous().view(-1).float()

            _safe_solve_quadprog._safe_patched = True
            _gem_module.GEMPlugin.solve_quadprog = _safe_solve_quadprog

        strategy_type = "Replay+Optimization"
        cl_strategy = GEM(
            model,
            optimizer,
            criterion,
            patterns_per_exp=config.strategy.patterns_per_exp,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "GDumb":
        strategy_type = "Replay"
        cl_strategy = GDumb(
            model,
            optimizer,
            criterion,
            mem_size=config.strategy.mem_size,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif "ICaRL" in strategy_name:
        strategy_type = "Replay"
        icarl_cls = ICaRL

        # Build a proper nn.Module feature extractor from the raw backbone.
        # WrapModel.feature_extractor is a method (not a module), which breaks
        # TrainEvalModel's nn.Module expectations. Instead, we create a thin
        # wrapper that calls forward_rep on the raw network.
        class FeatureExtractorWrapper(nn.Module):
            def __init__(self, raw_model):
                super().__init__()
                self.raw_model = raw_model

            def forward(self, x):
                return forward_rep(self.raw_model, x)

        raw_net = (
            model.model
        )  # The underlying ResNet inside WrapModel/WrapModelMT
        feature_extractor = FeatureExtractorWrapper(raw_net)

        cl_strategy = icarl_cls(
            feature_extractor=feature_extractor,
            classifier=fc_layer,
            optimizer=optimizer,
            memory_size=config.strategy.memory_size,
            fixed_memory=config.strategy.fixed_memory,
            buffer_transform=buffer_transform,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
        cl_strategy.model = TrainEvalModel(
            feature_extractor,
            fc_layer,
            eval_classifier=NCMClassifier(normalize=True),
            model=model,
            return_task_id=config.scenario.return_task_id,
        )

    elif strategy_name == "Replay":
        strategy_type = "Replay"
        cl_strategy = Replay(
            model,
            optimizer,
            criterion,
            mem_size=config.strategy.mem_size,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif "BiC" in strategy_name:
        strategy_type = "Replay"
        # stage2 epochs is set to train_epochs for simplicity, can be changed later
        cl_strategy = BiC(
            model,
            optimizer,
            criterion,
            mem_size=config.strategy.mem_size,
            val_percentage=getattr(config.strategy, "val_percentage", 0.1),
            T=getattr(config.strategy, "T", 2),
            stage_2_epochs=config.strategy.stage_2_epochs,
            lamb=getattr(config.strategy, "lamb", -1),
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "WA":
        strategy_type = "Replay+Regularization"
        wa_plugin = WeightAlignmentPlugin(
            mem_size=config.strategy.mem_size,
            temp=config.strategy.temp,
        )
        if plugins is None:
            plugins = [wa_plugin]
        else:
            plugins.append(wa_plugin)
        cl_strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif "FOSTER" in strategy_name:
        strategy_type = "Replay"
        foster_cfg = config.strategy
        foster_plugin = FOSTERPlugin(
            mem_size=foster_cfg.mem_size,
            beta1=float(getattr(foster_cfg, "beta1", 0.5)),
            beta2=float(getattr(foster_cfg, "beta2", 0.5)),
            lambda_okd=float(getattr(foster_cfg, "lambda_okd", 1.0)),
            T=float(getattr(foster_cfg, "T", 2.0)),
            compression_epochs=config.strategy.train_epochs,
            compression_lr=float(getattr(foster_cfg, "compression_lr", 0.1)),
            weight_decay=float(getattr(foster_cfg, "weight_decay", 2e-4)),
            compress_batch_size=int(
                getattr(
                    foster_cfg,
                    "compress_batch_size",
                    config.strategy.train_mb_size,
                )
            ),
            use_weight_align=bool(
                getattr(foster_cfg, "use_weight_align", False)
            ),
        )
        plugins.append(foster_plugin)
        cl_strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "mrfa":
        strategy_type = "Replay"
        mrfa_plugin = MRFAPlugin(
            mem_size=config.strategy.mem_size,
            net_type=config.model.name,
            beta=float(config.strategy.beta),
        )
        if plugins is None:
            plugins = [mrfa_plugin]
        else:
            plugins.append(mrfa_plugin)
        cl_strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    elif strategy_name == "PackNet":
        strategy_type = "architecture"
        # PackNet requires models without BatchNorm layers.
        cl_strategy = PackNet(
            model,
            optimizer,
            criterion,
            post_prune_epochs=config.strategy.post_prune_epochs,
            prune_proportion=config.strategy.prune_proportion,
            train_mb_size=config.strategy.train_mb_size,
            train_epochs=config.strategy.train_epochs,
            eval_mb_size=config.strategy.eval_mb_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    # PNN is intentionally excluded: Avalanche's implementation requires its
    # dedicated MLP-based PNN model rather than the shared backbone wrapper.
    else:
        print(f"Strategy '{strategy_name}' not recognized.")

    # Override Avalanche's default make_eval_dataloader to use num_workers>0.
    # Avalanche defaults to num_workers=0, which causes very slow eval on
    # ImageNet because JPEG decoding happens in the main process.
    if cl_strategy is not None:
        _orig_make_eval_dl = cl_strategy.make_eval_dataloader

        def _fast_make_eval_dataloader(self_unused=None, **kwargs):
            kwargs.setdefault("num_workers", 4)
            kwargs.setdefault("pin_memory", True)
            return _orig_make_eval_dl(**kwargs)

        import types

        cl_strategy.make_eval_dataloader = types.MethodType(
            lambda self, **kwargs: _fast_make_eval_dataloader(**kwargs),
            cl_strategy,
        )

    cl_strategy.model.apply(initialize_icarl_net)
    return cl_strategy, strategy_type


def get_transforms(config):
    dataset_name = config.dataset.name
    image_size = config.dataset.image_size
    train_transform = None
    eval_transform = None
    if dataset_name == "mnist":
        eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
                ),
            ]
        )
        train_transform = eval_transform
    elif dataset_name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),  # Resize first if needed, or crop from a padded image
                transforms.RandomCrop(
                    image_size, padding=4
                ),  # Standard CIFAR augmentation
                transforms.RandomHorizontalFlip(),  # Standard CIFAR augmentation
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # Remove: transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
    elif dataset_name == "cifar100":
        train_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),  # Resize first if needed, or crop from a padded image
                transforms.RandomCrop(
                    image_size, padding=4
                ),  # Standard CIFAR augmentation
                transforms.RandomHorizontalFlip(),  # Standard CIFAR augmentation
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
    elif dataset_name == "imagenet":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),  # Randomly crop and resize to the target size
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),  # Randomly adjust brightness, contrast, saturation, and hue
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return train_transform, eval_transform


def get_benchmark(config):
    # Ensure only rank 0 prepares the data to avoid race conditions in DDP
    distributed = is_dist_available_and_initialized()
    if distributed:
        rank = get_rank()
        if rank != 0:
            dist.barrier()

    dataset_name = config.dataset.name
    if hasattr(config.scenario, "fixed_class_order"):
        if (
            hasattr(config.scenario, "class_order_pickle")
            and os.path.isfile(config.scenario.class_order_pickle)
            and config.scenario.fixed_class_order != "fixed"
        ):
            with open(config.scenario.class_order_pickle, "rb") as f:
                fixed_class_order = pickle.load(f)
            fixed_class_order = fixed_class_order[
                config.scenario.fixed_class_order
            ][dataset_name]
        elif config.scenario.fixed_class_order == "random":
            # If fixed_class_order is set to random, we randomly permute the class order
            # for each run, but keep it consistent across runs.
            # This is useful for debugging and testing purposes.
            fixed_class_order = np.random.permutation(
                np.arange(config.dataset.num_classes)
            )
        elif config.scenario.fixed_class_order == "optimal":
            raise NotImplementedError(
                "You should pass a valid pickle file including the optimal order."
            )
        else:
            fixed_class_order = np.arange(config.dataset.num_classes)
    else:
        fixed_class_order = np.arange(config.dataset.num_classes)
    print(f"Using class order: {fixed_class_order}")
    train_transform, eval_transform = get_transforms(config)
    if dataset_name == "mnist":
        benchmark = SplitMNIST(
            eval_transform=eval_transform,
            train_transform=eval_transform,
            n_experiences=config.scenario.n_experiences,
            seed=config.scenario.seed,
            dataset_root=config.scenario.dataset_dir,
            shuffle=config.scenario.shuffle,
            return_task_id=config.scenario.return_task_id,
            fixed_class_order=fixed_class_order,
            class_ids_from_zero_in_each_exp=config.scenario.return_task_id,
            class_ids_from_zero_from_first_exp=not (
                config.scenario.return_task_id
            ),
        )
    elif dataset_name == "cifar10":
        # SplitCIFAR10 exposes the complete task-ID and class-remapping
        # controls used by this repository.
        benchmark = SplitCIFAR10(
            eval_transform=eval_transform,
            train_transform=train_transform,
            n_experiences=config.scenario.n_experiences,
            seed=config.scenario.seed,
            dataset_root=config.scenario.dataset_dir,
            shuffle=config.scenario.shuffle,
            fixed_class_order=fixed_class_order,
            return_task_id=config.scenario.return_task_id,
            class_ids_from_zero_in_each_exp=config.scenario.return_task_id,
            class_ids_from_zero_from_first_exp=not (
                config.scenario.return_task_id
            ),
        )
    elif dataset_name == "cifar100":
        benchmark = SplitCIFAR100(
            eval_transform=eval_transform,
            train_transform=train_transform,
            n_experiences=config.scenario.n_experiences,
            seed=config.scenario.seed,
            dataset_root=config.scenario.dataset_dir,
            shuffle=config.scenario.shuffle,
            return_task_id=config.scenario.return_task_id,
            fixed_class_order=fixed_class_order,
            class_ids_from_zero_in_each_exp=config.scenario.return_task_id,
            class_ids_from_zero_from_first_exp=not (
                config.scenario.return_task_id
            ),
        )
    elif dataset_name == "imagenet":
        benchmark = SplitImageNet(
            eval_transform=eval_transform,
            train_transform=train_transform,
            n_experiences=config.scenario.n_experiences,
            seed=config.scenario.seed,
            dataset_root=config.scenario.dataset_dir,
            shuffle=config.scenario.shuffle,
            return_task_id=config.scenario.return_task_id,
            fixed_class_order=fixed_class_order,
            class_ids_from_zero_in_each_exp=config.scenario.return_task_id,
            class_ids_from_zero_from_first_exp=not (
                config.scenario.return_task_id
            ),
        )
    else:
        raise ValueError("Invalid dataset name {}".format(config.dataset.name))

    if distributed:
        if rank == 0:
            dist.barrier()

    return benchmark, eval_transform


def fix_inplace_relu(model):
    # fixing inplace operation error cause by relu
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            # Split the name to access the parent module
            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                # Get the parent module
                parent_module = dict(model.named_modules())[parent_name]
                # Get the attribute name (e.g., 'relu')
                relu_attr_name = name.rsplit(".", 1)[1]
            else:
                parent_module = model
                relu_attr_name = name
            # Replace the ReLU layer
            setattr(parent_module, relu_attr_name, nn.ReLU(inplace=False))


def init_fc_layer(config, in_features, out_features):
    is_joint = config.strategy.name == "Joint"
    if is_joint:
        return nn.Linear(in_features, out_features)

    use_cosine_clf = (
        hasattr(config.scenario, "use_cosine_clf")
        and config.scenario.use_cosine_clf
    )
    if use_cosine_clf and config.scenario.return_task_id:
        fc_layer = CosineMultiHeadClassifier(
            in_features, initial_out_features=out_features
        )
    elif use_cosine_clf:
        fc_layer = CosineIncrementalClassifier(in_features, out_features)
    elif config.scenario.return_task_id:
        fc_layer = MultiHeadClassifier(
            in_features, initial_out_features=out_features
        )
    else:
        fc_layer = IncrementalClassifier(in_features, out_features)
    return fc_layer


def init_model(config, benchmark):
    postprocessor_name = config.postprocessor.name
    image_size = config.dataset.image_size
    is_joint = config.strategy.name == "Joint"
    if is_joint:
        init_n_classes = sum(
            [
                len(benchmark._classes_in_exp[i])
                for i in range(config.scenario.n_experiences)
            ]
        )
    else:
        init_n_classes = len(benchmark._classes_in_exp[0])

    return_task_id = config.scenario.return_task_id
    model_name = config.model.name
    fc_layer = None
    if model_name == "mlp":
        model = SimpleMLP(
            num_classes=config.dataset.num_classes, input_size=3 * image_size**2
        )
        fc_layer = init_fc_layer(
            config, model.classifier.in_features, init_n_classes
        )
        model.classifier = fc_layer
    elif model_name == "mobilenet":
        model = MobilenetV1(pretrained=False)
        fc_layer = init_fc_layer(config, 1024, init_n_classes)
        model.output = fc_layer
    elif model_name == "resnet50":
        model = ResNet50(num_classes=config.dataset.num_classes)
        fc_layer = init_fc_layer(config, model.fc.in_features, init_n_classes)
        model.fc = fc_layer
    elif model_name == "resnet18_224":
        model = ResNet18_224x224(num_classes=config.dataset.num_classes)
        fc_layer = init_fc_layer(config, model.fc.in_features, init_n_classes)
        model.fc = fc_layer
    elif model_name == "resnet32":
        model = make_icarl_net(num_classes=config.dataset.num_classes)
        fc_layer = init_fc_layer(
            config, model.classifier.in_features, init_n_classes
        )
        model.fc = fc_layer
    elif model_name == "resnet18":
        model = ResNet18_32x32(num_classes=config.dataset.num_classes)
        fc_layer = init_fc_layer(config, model.fc.in_features, init_n_classes)
        model.fc = fc_layer
    elif model_name == "vit":
        model = ViT_B_16(image_size, num_classes=config.dataset.num_classes)
        # Load the ImageNet-1k pretrained backbone when requested. Previously
        # `network.pretrained` was a no-op for ViT, so the transformer trained
        # from random init -> catastrophic on small images. We load torchvision's
        # ViT-B/16 IMAGENET1K_V1 weights (224x224, 197 pos tokens — matches this
        # config) and drop the 1000-way head (the CL head is built below).
        if getattr(config.network, "pretrained", False):
            import torchvision.models as _tv

            sd = _tv.vit_b_16(
                weights=_tv.ViT_B_16_Weights.IMAGENET1K_V1
            ).state_dict()
            sd = {k: v for k, v in sd.items() if not k.startswith("heads.")}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(
                f"[ViT] loaded ImageNet-1k pretrained backbone "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        fc_layer = init_fc_layer(
            config, model.heads.head.in_features, init_n_classes
        )
        model.heads = fc_layer
    else:
        raise ValueError("Invalid model name {}".format(model_name))
    fix_inplace_relu(model)
    # Keep initialization policy model-specific. The iCaRL ResNet32 expects the
    # Avalanche iCaRL initializer; other backbones should keep their constructor
    # defaults rather than being reinitialized with that scheme.
    if model_name == "resnet32":
        model.apply(initialize_icarl_net)
    if config.strategy.name == "PackNet":
        model = PackNetModel(model)

    if return_task_id:
        model = WrapModelMT(model, fc_layer)
    else:
        model = WrapModel(model, fc_layer)
    # ASH and ReAct currently operate through the shared single-head wrapper;
    # multi-head support would require a specialized classifier wrapper.
    if postprocessor_name == "ash":
        model.enable_ash = True
    elif "adascale" in postprocessor_name:
        model.enable_adascale = True
        model.logit_scaling = "adascale_l" == postprocessor_name
    return model, fc_layer
