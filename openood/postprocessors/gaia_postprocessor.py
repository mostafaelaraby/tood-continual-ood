from typing import Any
import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


# All hooks need data type
class Grad_all_hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_grad)
        self.data = torch.Tensor()

    def save_grad(self, module, input, output):
        def _stor_grad(grad):
            self.data = grad.detach()

        output.register_hook(_stor_grad)

    def close(self):
        self.hook.remove()


class Activation_all_hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_activations)
        self.data = torch.Tensor()

    def save_activations(self, module, input, output):
        self.data = output.detach()

    def close(self):
        self.hook.remove()


class Grad_feature_hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_grad)
        self.data = torch.Tensor()
        self.feature = torch.Tensor()

    def save_grad(self, module, input, output):
        def _stor_grad(grad):
            self.data = grad.detach()

        output.register_hook(_stor_grad)
        self.feature = output.clone()

    def close(self):
        self.hook.remove()


def get_bn_hooks(net):
    bn_hooks = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.GroupNorm):
            bn_hooks.append(Grad_all_hook(module))
    return bn_hooks


def get_beforehead_hooks(net):
    """
    Gets hooks for modules before the classification head, optimized and model-agnostic.

    Args:
        net: The PyTorch model.

    Returns:
        A list of hooks.
    """
    beforehead_hooks = []
    module_list = []

    # Helper function to recursively find modules of a specific type
    def find_modules(module, module_type, target_list):
        for submodule in module.children():  # Use .children(), not .modules()
            if isinstance(submodule, module_type):
                target_list.append(submodule)
            find_modules(submodule, module_type, target_list)  # Recurse

    # --- ResNet-like structures (resnet18, resnet34, resnet50, etc.) ---
    if hasattr(net, "layer3") and hasattr(net, "layer4"):
        find_modules(net.layer3, nn.BatchNorm2d, module_list)
        find_modules(net.layer4, nn.BatchNorm2d, module_list)
        module_list.append(net.layer4)  # Add the entire layer4

    # --- Wide ResNet (wrn_40_2, etc.) ---
    elif hasattr(net, "AdaptAvgPool"):  # Check for a characteristic attribute
        find_modules(net, nn.BatchNorm2d, module_list)
        module_list.append(net.AdaptAvgPool)

    # --- VGG-like structures (vgg, vgg16, etc.) ---
    elif hasattr(net, "pool4"):  # Use 'pool4' as a characteristic
        find_modules(net, nn.BatchNorm2d, module_list)
        module_list.append(net.pool4)

    # --- Fallback (if no specific structure is detected) ---
    else:
        # You might want to log a warning here, as this is a less specific case
        print(
            "Warning: Model structure not explicitly handled.  Using generic module search."
        )
        find_modules(
            net, (nn.BatchNorm2d, nn.GroupNorm), module_list
        )  # Search for common normalization layers
        #  Consider adding a final layer/module if you know *something* about the structure

    # Create hooks (assuming Grad_all_hook and Grad_feature_hook are defined)
    for index, module in enumerate(module_list):
        if index < len(module_list) - 1:
            beforehead_hooks.append(Grad_all_hook(module))
        else:
            beforehead_hooks.append(Grad_feature_hook(module))
    return beforehead_hooks


def cal_grad_value(net, data, hooks=None):
    net.zero_grad()
    bsz = data.shape[0]
    y, features = net(data, return_feature_list=True)
    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(data.device)

    loss = logsoftmax(y)
    loss.sum().backward(retain_graph=True)
    before_head_grad = hooks[-1].data.mean(dim=(-1, -2))
    output_component = torch.sqrt(torch.abs(before_head_grad).mean(dim=1))
    output_component = output_component.unsqueeze(dim=1)

    loss = features[-1].view(bsz, -1)
    loss.sum().backward()
    gradients = [hook.data for hook in hooks]
    gradients = gradients[:-1]
    gradients = [grad.mean(dim=(-1, -2)) for grad in gradients]
    inner_component = torch.abs(torch.cat(gradients, dim=1))
    score = torch.pow(inner_component / output_component, 2).mean(dim=1)
    return score.detach(), y.argmax(1)


class GaiaPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.net = net
        # TODO check if the model has bn layers or group norm ones
        self.hooks = get_bn_hooks(self.net)
        # self.hooks = get_beforehead_hooks(self.net)

    @torch.enable_grad()
    def postprocess(self, net: nn.Module, data: Any):
        scores, preds = cal_grad_value(self.net, data, self.hooks)
        return preds, -1 * scores
