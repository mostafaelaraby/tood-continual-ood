from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ExperienceBalancedBuffer
from torch.utils.data import default_collate

# =============================================================================
# 1. MRFA Logic (Core Algorithm)
# =============================================================================


class MRFA:
    """
    Holds:
      - computed perturbation directions per layer (self.perturbations)
      - per-minibatch instructions: which sample gets perturbed at which layer,
        and with what magnitude
      - hook handles for cleanup
    """

    def __init__(self, with_input_norm: bool = True):
        self.with_input_norm = with_input_norm
        self.perturbations = []  # list[Tensor], one per hooked layer
        self.remove_handles = []
        self._init_inbatch_properties()

    def _init_inbatch_properties(self):
        # Per-minibatch perturbation instructions
        self.perturbation_layers = []  # list[int]
        self.perturbation_factor = []  # list[float]
        self.perturbation_idices = (
            []
        )  # list[int] (index into stored perturbations batch dim)
        self.perturbation_idices_inbatch = (
            []
        )  # list[int] (index into actual forward batch)

    def feature_augmentation(self, model, samples, targets, net_type: str):
        """
        Computes gradient ascent directions (w.r.t. selected layer inputs)
        for the given (replay) samples.
        """
        if net_type == "resnet32":
            self.get_feature_augmentation(
                model=model,
                convnet=model.model,
                samples=samples,
                targets=targets,
                num_layers=4,
                register_func=register_forward_prehook_resnet32,
            )
        elif net_type == "resnet18":
            target_net = getattr(model.model, "convnet", model.model)
            self.get_feature_augmentation(
                model=model,
                convnet=target_net,
                samples=samples,
                targets=targets,
                num_layers=5,
                register_func=register_forward_prehook_resnet18,
            )
        elif net_type == "mobilenet":
            target_net = getattr(model.model, "convnet", model.model)
            self.get_feature_augmentation(
                model=model,
                convnet=target_net,
                samples=samples,
                targets=targets,
                num_layers=3,
                register_func=register_forward_prehook_mobilenet,
            )
        else:
            raise ValueError(f"Unknown net_type {net_type}.")

    def get_feature_augmentation(
        self, model, convnet, samples, targets, num_layers: int, register_func
    ):
        """
        Register "capture input" prehooks -> forward -> backward to obtain grads
        for each hooked layer input tensor.
        """
        layer_inputs = []

        def get_input_prehook(module, inp):
            # inp is a tuple; inp[0] is the Tensor
            inp[0].retain_grad()
            layer_inputs.append(inp[0])

        # Register capture hooks
        remove_handles = register_func(
            model, convnet, [get_input_prehook] * num_layers
        )

        # Need grads on samples path
        samples.requires_grad_(True)

        prev_training = model.training
        model.eval()  # freeze BN stats during this aux pass

        outputs = model(samples)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        loss = F.cross_entropy(logits, targets)

        model.zero_grad(set_to_none=True)
        loss.backward()

        # Store perturbation directions (grads w.r.t. layer inputs)
        self.perturbations = []
        for inp in layer_inputs:
            if inp.grad is None:
                self.perturbations.append(torch.zeros_like(inp))
            else:
                self.perturbations.append(inp.grad.detach().clone())

        # Cleanup
        model.zero_grad(set_to_none=True)
        samples.requires_grad_(False)
        for h in remove_handles:
            h.remove()
        model.train(prev_training)

    def register_perturb_forward_prehook(self, model, net_type: str):
        """
        Registers the hooks that APPLY the calculated perturbations during training.
        """
        if net_type == "resnet32":
            self._register_perturb_forward_prehook_layers(
                model, model.model, 4, register_forward_prehook_resnet32
            )
        elif net_type == "resnet18":
            target_net = getattr(model.model, "convnet", model.model)
            self._register_perturb_forward_prehook_layers(
                model, target_net, 5, register_forward_prehook_resnet18
            )
        elif net_type == "mobilenet":
            target_net = getattr(model.model, "convnet", model.model)
            self._register_perturb_forward_prehook_layers(
                model, target_net, 3, register_forward_prehook_mobilenet
            )
        else:
            raise ValueError(f"Unknown net_type {net_type}.")

    def _register_perturb_forward_prehook_layers(
        self, model, convnet, num_layers: int, register_func
    ):
        """
        Core hook that mutates layer input for selected replay samples.
        Fixes:
          - robust float conversion (prevents numpy.str_ failure)
          - avoids repeated np.array creation with ambiguous dtype
        """

        def perturb_input_prehook_full(module, inp, layer_id: int):
            if layer_id not in self.perturbation_layers:
                return inp

            inp0 = inp[0].clone()

            # Build numeric arrays robustly (avoid numpy.str_ issues)
            p_layers = np.asarray(self.perturbation_layers, dtype=np.int64)
            p_indices = np.asarray(self.perturbation_idices, dtype=np.int64)
            p_indices_inbatch = np.asarray(
                self.perturbation_idices_inbatch, dtype=np.int64
            )

            # Factors must be float32; if anything is a string, this coercion fixes it
            p_factor = np.asarray(self.perturbation_factor, dtype=np.float32)

            mask = p_layers == int(layer_id)
            if not np.any(mask):
                return (inp0,)

            batch_idxs = p_indices_inbatch[mask]
            perturb_idxs = p_indices[mask]

            # Convert factors to torch (already float32)
            current_factors = torch.from_numpy(p_factor[mask]).to(
                device=inp0.device, dtype=torch.float32
            )

            perturb_tensor = self.perturbations[
                layer_id
            ]  # shape [B_replay, ...]
            grad_term = perturb_tensor[perturb_idxs]

            # Broadcast factors to match grad tensor dims
            num_dims = grad_term.dim()
            view_shape = [-1] + [1] * (num_dims - 1)
            current_factors = current_factors.view(*view_shape)

            if self.with_input_norm:
                # Scale by squared L2 norm of the layer input feature, per sample
                flat_input = inp0[batch_idxs].view(len(batch_idxs), -1)
                norm = flat_input.norm(dim=-1, keepdim=True).view(*view_shape)
                perturb = norm * grad_term * current_factors
            else:
                perturb = grad_term * current_factors

            inp0[batch_idxs] = inp0[batch_idxs] + perturb
            return (inp0,)

        hooks = [
            partial(perturb_input_prehook_full, layer_id=i)
            for i in range(num_layers)
        ]
        self.remove_handles.extend(register_func(model, convnet, hooks))

    def cleanup_hooks(self):
        for h in self.remove_handles:
            h.remove()
        self.remove_handles = []


# =============================================================================
# 2. Avalanche Plugin (Main Interface)
# =============================================================================


class MRFAPlugin(SupervisedPlugin):
    """
    Multi-layer Rehearsal Feature Augmentation (MRFA) Plugin.

    Fixes vs your current version:
      - randomized scaling: beta_hat ~ U(0, beta) per replay sample
      - robust float conversion so numpy.str_ cannot crash torch.from_numpy
      - removes pdb breakpoint
      - uses MRFA._init_inbatch_properties() for consistent resets
    """

    def __init__(
        self,
        mem_size: int = 2000,
        net_type: str = "resnet32",
        beta: float = 1e-3,  # paper-style max scale
        layers=(0, 1, 2, 3),  # which layer IDs are eligible
        with_input_norm: bool = True,
        update_freq: int = 1,
    ):
        super().__init__()
        self.mrfa = MRFA(with_input_norm=with_input_norm)
        self.net_type = net_type
        self.beta = float(beta)
        self.target_layers = list(layers)
        self.counter = 0
        self.update_freq = update_freq  # or 2, 5, etc.
        self.storage_policy = ExperienceBalancedBuffer(
            max_size=int(mem_size), adaptive_size=True
        )

        # bookkeeping for current iteration replay slice
        self._replay_start_idx = None
        self._replay_end_idx = None

    def before_training_exp(self, strategy, **kwargs):
        # Update buffer at start of each experience (same as your current code)
        self.storage_policy.post_adapt(strategy, strategy.experience)

    def before_training_iteration(self, strategy, **kwargs):
        """
        Mix replay data into the current batch.
        """
        if len(self.storage_policy.buffer) == 0:
            self._replay_start_idx = None
            self._replay_end_idx = None
            return

        batch_size = strategy.mb_x.size(0)

        buffer_size = len(self.storage_policy.buffer)
        indices = torch.randint(0, buffer_size, (batch_size,)).tolist()
        batch_samples = [self.storage_policy.buffer[i] for i in indices]
        replay_batch = default_collate(batch_samples)

        replay_x, replay_y, replay_t = (
            replay_batch[0],
            replay_batch[1],
            replay_batch[2],
        )
        replay_x = replay_x.to(strategy.device)
        replay_y = replay_y.to(strategy.device)
        replay_t = replay_t.to(strategy.device)

        self._replay_start_idx = strategy.mb_x.size(0)
        replay_len = replay_x.size(0)
        self._replay_end_idx = self._replay_start_idx + replay_len

        new_x = torch.cat([strategy.mb_x, replay_x], dim=0)
        new_y = torch.cat([strategy.mb_y, replay_y], dim=0)
        new_t = torch.cat([strategy.mb_task_id, replay_t], dim=0)

        strategy.mbatch = (new_x, new_y, new_t)

    def before_forward(self, strategy, **kwargs):
        """
        1) Compute MRFA gradients on the replay subset
        2) Configure per-sample perturbation instructions with randomized scale
        3) Register perturbation hooks for this forward only
        """
        # Only apply if replay is present AND not first experience (as in your code)
        if self._replay_start_idx is None:
            return
        if getattr(strategy.experience, "current_experience", 0) == 0:
            return
        self.counter += 1
        if self.counter % self.update_freq != 0:
            return
        replay_indices = torch.arange(
            self._replay_start_idx, self._replay_end_idx, device=strategy.device
        )
        x_replay = strategy.mbatch[0][replay_indices]
        y_replay = strategy.mbatch[1][replay_indices]

        # Step 1: compute gradient-ascent directions for replay samples
        self.mrfa.feature_augmentation(
            strategy.model, x_replay, y_replay, self.net_type
        )

        # Step 2: build perturbation instructions (reset consistently)
        self.mrfa._init_inbatch_properties()

        num_replay = int(replay_indices.numel())
        if num_replay == 0:
            return

        # Uniformly choose a target layer per replay sample
        # (target_layers is a subset mapping into your layer_id space)
        rand_layer_pos = torch.randint(
            low=0,
            high=len(self.target_layers),
            size=(num_replay,),
            device=strategy.device,
        )
        chosen_layers = [
            int(self.target_layers[i]) for i in rand_layer_pos.tolist()
        ]

        # Randomized scale per sample: beta_hat ~ U(0, beta)
        # Store as Python floats to keep serialization simple, but numeric.
        beta_hat = (
            torch.rand(num_replay, device=strategy.device) * self.beta
        ).tolist()

        # Indices mapping:
        # - perturbation_idices: index into MRFA.perturbations batch dimension (0..num_replay-1)
        # - perturbation_idices_inbatch: actual index in current strategy.mbatch
        for i in range(num_replay):
            self.mrfa.perturbation_layers.append(chosen_layers[i])
            self.mrfa.perturbation_factor.append(
                float(beta_hat[i])
            )  # force numeric
            self.mrfa.perturbation_idices.append(i)
            self.mrfa.perturbation_idices_inbatch.append(
                int(replay_indices[i].item())
            )

        # Step 3: register prehooks to apply perturbations during forward
        self.mrfa.register_perturb_forward_prehook(
            strategy.model, self.net_type
        )

    def after_forward(self, strategy, **kwargs):
        """
        Remove hooks immediately so eval/validation/test are not affected.
        """
        self.mrfa.cleanup_hooks()

        # Clear replay slice bookkeeping
        self._replay_start_idx = None
        self._replay_end_idx = None


# =============================================================================
# 3. Hook Registration Helpers (keep your existing ones)
# =============================================================================


def register_forward_prehook_resnet32(model, convnet, hooks):
    """
    Hooks for your ResNet32.
    Expects convnet has: layer1, layer2, layer3; and model has fc optionally.
    """
    remove_handles = []
    layers = [convnet.layer1, convnet.layer2, convnet.layer3]

    for i, layer in enumerate(layers):
        if i < len(hooks):
            remove_handles.append(layer.register_forward_pre_hook(hooks[i]))

    if len(hooks) > 3 and hasattr(model, "fc"):
        remove_handles.append(model.fc.register_forward_pre_hook(hooks[3]))

    return remove_handles


def register_forward_prehook_resnet18(model, convnet, hooks):
    """
    Hooks for torchvision ResNet18.
    """
    remove_handles = []
    layers = [convnet.layer1, convnet.layer2, convnet.layer3, convnet.layer4]

    for i, layer in enumerate(layers):
        if i < len(hooks):
            remove_handles.append(layer.register_forward_pre_hook(hooks[i]))

    if len(hooks) > 4 and hasattr(model, "fc"):
        remove_handles.append(model.fc.register_forward_pre_hook(hooks[4]))

    return remove_handles


def register_forward_prehook_mobilenet(model, convnet, hooks):
    """
    Keep your existing MobileNet hook selection logic here.
    (I am not changing it, because it is model-definition-specific.)
    """
    remove_handles = []

    targets = []

    if len(model.lat_features) > 0:
        targets.append(model.lat_features[0])
    if len(model.end_features) > 0:
        targets.append(model.end_features[0])
    if hasattr(model, "output"):
        targets.append(model.output)

    if len(hooks) > len(targets) and len(model.lat_features) > 10:
        mid_idx = len(model.lat_features) // 2
        targets.insert(1, model.lat_features[mid_idx])

    for i, layer in enumerate(targets):
        if i < len(hooks):
            remove_handles.append(layer.register_forward_pre_hook(hooks[i]))

    return remove_handles
