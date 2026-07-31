from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from utils.helpers import get_fc_layer


class InferenceHook:
    def __init__(self, module):
        self.module = module
        self.hook = None
        self.features = None
        self.logits = None

    def hook_fn(self, module, input, output):
        # input is a tuple (features,), output is logits
        self.features = input[0]
        self.logits = output

    def __enter__(self):
        net = self.module
        if hasattr(net, "module"):
            net = net.module
        target_layer = get_fc_layer(net)
        if isinstance(target_layer, list):
            target_layer = target_layer[0]
        if target_layer is not None:
            self.hook = target_layer.register_forward_hook(self.hook_fn)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook:
            self.hook.remove()
        self.features = None
        self.logits = None


class BasePostprocessor:
    def __init__(self, config):
        self.config = config
        self.temperature = 1.0

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def _get_logits(self, net, loader, max_samples=None):
        """
        Helper to extract raw logits.
        Stops early if max_samples is reached.
        """
        logits_list = []
        total_samples = 0

        # Use tqdm only if it's the main process to avoid log clutter
        iterator = tqdm(
            loader, disable=not comm.is_main_process(), desc="Extracting Logits"
        )

        for batch in iterator:
            data = batch["data"].cuda()
            logits = net(data)
            logits_list.append(logits)

            total_samples += logits.size(0)
            if max_samples is not None and total_samples >= max_samples:
                break

        all_logits = torch.cat(logits_list)

        # Trim exact amount if we overshot a bit due to batch size
        if max_samples is not None and all_logits.size(0) > max_samples:
            all_logits = all_logits[:max_samples]

        return all_logits

    def get_hyperparams(self):
        return {"temperature": self.temperature}

    def set_hyperparams(self, params: dict):
        if "temperature" in params:
            self.temperature = params["temperature"]

    def tune_params(
        self,
        net: nn.Module,
        id_loader,
        ood_loader,
        max_samples=None,
    ) -> None:
        """
        Tunes the temperature parameter to maximize AUROC between
        ID (buffer) and OOD (validation) data.
        """

        # If a subclass calls this method but did not override it,
        # notify and return early.
        cls_post = getattr(self.__class__, "postprocess", None)
        if (
            cls_post is BasePostprocessor.postprocess
            and self.__class__ is not BasePostprocessor
        ):
            print(
                f"[{self.__class__.__name__}] postprocess was not overridden; returning.",
                flush=True,
            )
            return None
        print(
            f"[{self.__class__.__name__}] Starting hyperparameter tuning...",
            flush=True,
        )

        # 1. Extract Logits (Do this once to save time)
        id_logits = self._get_logits(net, id_loader, max_samples=max_samples)
        ood_logits = self._get_logits(net, ood_loader, max_samples=max_samples)

        # 2. Define Grid Search Space for Temperature
        # We look at standard temp scaling ranges, plus some aggressive ones
        t_list = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0]

        best_t = 1.0
        best_auroc = 0.0

        # 3. Grid Search
        for t in t_list:
            # Calculate scores with current T
            id_scores = (
                torch.softmax(id_logits / t, dim=1).max(dim=1)[0].cpu().numpy()
            )
            ood_scores = (
                torch.softmax(ood_logits / t, dim=1).max(dim=1)[0].cpu().numpy()
            )

            # Prepare labels for AUROC (ID=1, OOD=0)
            scores = np.concatenate([id_scores, ood_scores])
            labels = np.concatenate(
                [np.ones_like(id_scores), np.zeros_like(ood_scores)]
            )

            # Calculate AUROC
            try:
                auroc = roc_auc_score(labels, scores)
            except ValueError:
                print(
                    "[Warning] Not enough samples to compute AUROC.", flush=True
                )
                auroc = 0.0  # Handle edge cases with not enough samples

            if auroc > best_auroc:
                best_auroc = auroc
                best_t = t

            # Optional: Verbose logging
            # print(f"  Temp: {t}, AUROC: {auroc:.4f}", flush=True)

        # 4. Update the parameter
        self.temperature = best_t
        print(
            f"[{self.__class__.__name__}] Tuning complete. Best Temp: {self.temperature} (AUROC: {best_auroc:.4f})",
            flush=True,
        )

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output / self.temperature, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(
        self,
        net: nn.Module,
        data_loader: DataLoader,
        progress: bool = True,
        return_logits: bool = False,
        return_feature: bool = False,
    ):
        pred_list, conf_list, label_list, logits_list, features_list = (
            [],
            [],
            [],
            [],
            [],
        )
        # Forward hooks retain classifier inputs/outputs and add a Python
        # callback to every batch. Install one only when the caller actually
        # requests logits or features.
        hook_context = (
            InferenceHook(net)
            if return_logits or return_feature
            else nullcontext(None)
        )
        with hook_context as hook:
            for batch in tqdm(
                data_loader, disable=not progress or not comm.is_main_process()
            ):
                data = batch["data"].cuda(non_blocking=True)
                # Labels are only returned to the CPU; transferring them to
                # CUDA and immediately back adds needless synchronization.
                label = batch["label"]
                pred, conf = self.postprocess(net, data)
                features = hook.features if hook is not None else None
                logits = hook.logits if hook is not None else None
                if (logits is None and return_logits) or (
                    features is None and return_feature
                ):
                    # a dirty fix to avoid issues
                    logits, features = net(data, return_feature=True)
                pred_list.append(pred.cpu())
                conf_list.append(conf.cpu())
                label_list.append(label.cpu())
                if return_logits:
                    logits_list.append(logits.cpu().detach())
                if return_feature:
                    features_list.append(features.cpu().detach())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        logits_list = torch.cat(logits_list).numpy() if return_logits else None
        features_list = (
            torch.cat(features_list).numpy() if return_feature else None
        )
        if return_logits and return_feature:
            return pred_list, conf_list, label_list, logits_list, features_list
        if return_logits:
            return pred_list, conf_list, label_list, logits_list
        if return_feature:
            return pred_list, conf_list, label_list, features_list

        return pred_list, conf_list, label_list
