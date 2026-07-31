from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm
from .base_postprocessor import BasePostprocessor
import torch.nn.functional as F


def preprocess_batch(batch):
    if (
        type(batch) is dict
    ):  # TODO: Make this condition permanent, when all dataloaders follow OpenOOD standards
        batch = batch["data"], batch["label"]
    batch = batch[0].cuda().float(), batch[1].cuda().float()
    return batch

# Ngoc-Hieu, Nguyen, et al. "A Cosine Similarity-based Method for Out-of-Distribution Detection." arXiv preprint arXiv:2306.14920 (2023).
class CosinePostProcessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.args = config.postprocessor.postprocessor_args
        self.sampling_ratio = self.args.sampling_ratio
        self._prototypes_tensors = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        train_dataloader = id_loader_dict["train"]
        rand_data = next(iter(train_dataloader))
        rand_data = preprocess_batch(rand_data)
        _, features = net(rand_data[0], return_feature_list=True)
        penultimate_dim = features[-1].view(rand_data[0].shape[0], -1).shape[-1]
        # compute prototypes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = self.config.dataset.num_classes
        self._prototypes_tensors = torch.zeros(
            [num_classes, penultimate_dim],
            device=self.device,
            requires_grad=False,
        )
        self._count_features = torch.zeros(
            [num_classes],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self._prototypes_tensors.requires_grad = False
        self._count_features.requires_grad = False
        self.network = net
        if not (self.are_prototypes_ready()):
            # n_samples = self.sampling_ratio * len(train_dataloader)
            n_samples = 25
            for idx, batch in tqdm(
                enumerate(train_dataloader),
                desc="Eval: update prototype: ",
                position=0,
                leave=True,
            ):
                batch = preprocess_batch(batch)
                self.update_prototype(batch)
                if idx >= n_samples:
                    break

    @property
    def prototypes(self):
        return self._prototypes_tensors

    def are_prototypes_ready(self):
        """Detects if at least a data point per class seen so far
        Returns:
            bool: flag to denote if all our prototypes are nonzeros
        """
        return (
            self._count_features is not None
            and self._count_features.count_nonzero() == self._count_features.shape[0]
        )

    def update_prototype(self, batch):
        def get_total_features(feats_cl, cl, proto_tensor):
            feats_cl_sum = torch.sum(feats_cl, dim=0)
            return (feats_cl_sum + self._count_features[cl] * proto_tensor[cl]) / (
                self._count_features[cl] + n_features
            )

        data, target = batch
        logits, features = self.network(data, return_feature_list=True)
        pseudo_targets = logits.argmax(1)
        # for the prototypes
        self.penultimate_dim = features[-1].shape[1]
        features = features[-1].detach().reshape(data.shape[0], -1)
        classes = target.unique()
        for cl in classes:
            cl = cl.long()
            masked_cl = (target == cl).bool() & (pseudo_targets == cl).bool()
            n_features = masked_cl.sum()
            if n_features == 0:
                continue
            self._prototypes_tensors[cl] = get_total_features(
                features[masked_cl], cl, self._prototypes_tensors
            )
            self._count_features[cl] += n_features

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        bsz = data.shape[0]
        logits, features = net(data, return_feature_list=True)
        softmax_scores = torch.softmax(logits, dim=1)
        _, pred = torch.max(softmax_scores, dim=1)
        # now computing the confidence based on thecosine distance from the corresponding prototype
        conf = F.cosine_similarity(
            features[-1].detach().view(bsz, -1), self.prototypes.unsqueeze(1), dim=2
        ).t()
        # now select based on the pred 
        conf = conf[torch.arange(bsz), pred.long()]
        return pred, conf
