import torch
import torch.nn as nn
from avalanche.models import MultiTaskModule
from avalanche.models import TrainEvalModel as TrainEvalModelAVL

from openood.networks.adascale_net import ada_scale
from openood.networks.ash_net import ash_b
from utils.helpers import forward_rep, get_fc_layer, get_fc_w_b


def clip_features(model, features, threshold):
    scale = None
    if model.enable_ash:
        bsz = features.size(0)
        features = ash_b(features.view(bsz, -1, 1, 1), threshold)
        features = features.view(bsz, -1)
    elif model.enable_adascale:
        scale = ada_scale(torch.relu(features), threshold)
        features = (
            features if model.logit_scaling else features * torch.exp(scale)
        )
    else:
        features = features.clip(max=threshold)
    return features, scale


class TrainEvalModel(TrainEvalModelAVL, MultiTaskModule):
    def __init__(
        self,
        feature_extractor,
        train_classifier,
        eval_classifier,
        model=None,
        return_task_id=False,
    ):
        super().__init__(feature_extractor, train_classifier, eval_classifier)
        self.model = model
        self.return_task_id = return_task_id
        eval_classifier.return_task_id = return_task_id
        self._task_id = None
        self.num_classes = None
        self.feature_size = None
        self.return_combined_logits = False
        self._is_ood_eval = False

    @property
    def is_ood_eval(self):
        return self._is_ood_eval

    @is_ood_eval.setter
    def is_ood_eval(self, v):
        self._is_ood_eval = v

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, v):
        self.model.task_id = v
        self._task_id = v
        try:
            self.eval_classifier.task_id = v
            if hasattr(self.train_classifier, "classifiers"):
                self.eval_classifier.n_classes_per_task = (
                    self.train_classifier.classifiers[
                        str(v)
                    ].classifier.out_features
                )
            elif hasattr(self.train_classifier, "classifier"):
                self.eval_classifier.n_classes_per_task = int(
                    self.train_classifier.classifier.out_features
                    // self.task_id
                )
            else:
                self.eval_classifier.n_classes_per_task = int(
                    self.train_classifier.out_features // self.task_id
                )
        except Exception:
            pass

    def toggle_combined_logits(self):
        self.return_combined_logits = not (self.return_combined_logits)
        self.model.toggle_combined_logits()

    def get_features(self, x):
        return self.feature_extractor(x)

    def forward(
        self,
        x,
        task_labels=None,
        return_feature=False,
        return_feature_list=False,
        threshold=None,
    ):
        requires_grad = x.requires_grad
        features = self.feature_extractor(x)
        scale = None
        if threshold is not None:
            features, scale = clip_features(self.model, features, threshold)
        if scale is not None and getattr(self.model, "logit_scaling", False):
            scale = scale**2.0
        else:
            scale = 1.0
        if self.feature_size is None:
            self.feature_size = features.view(x.shape[0], -1).size(1)
        if task_labels is None and self.task_id is not None:
            task_labels = self.task_id
        if self.training or requires_grad or self.is_ood_eval:
            logits = self.train_classifier(
                features,
                task_labels=(
                    task_labels if not (self.return_combined_logits) else None
                ),
            )
        else:
            logits = self.eval_classifier(features)
        if self.return_combined_logits and isinstance(logits, dict):
            # combine all logits into a single tensor
            combined_logits = []
            for task_id in sorted(logits.keys()):
                combined_logits.append(logits[task_id])
            logits = torch.cat(combined_logits, dim=1)
        if isinstance(logits, torch.Tensor):
            logits = logits * scale
        elif isinstance(logits, dict):
            for key in logits:
                logits[key] = logits[key] * scale
        if return_feature:
            return logits, features
        if return_feature_list:
            return logits, [features]
        return logits

    def forward_threshold(self, x, threshold=None):
        return self.model.forward_threshold(x, threshold=threshold)

    def get_fc(self):
        """
        Returns the fully connected layer of the model.
        This is useful for strategies that require access to the classifier layer.
        """
        fc = self.get_fc_layer()
        return get_fc_w_b(fc, self.num_classes)

    def get_fc_layer(self):
        """
        Returns the fully connected layer of the model.
        This is useful for strategies that require access to the classifier layer.
        """
        if hasattr(self.train_classifier, "get_fc_layer"):
            return self.train_classifier.get_fc_layer()
        return get_fc_layer(self.train_classifier, task_id=self.task_id)


class WrapModel(nn.Module):
    def __init__(self, model, classifier, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.classifier = classifier
        self.task_id = None
        self.enable_ash = False
        # adscale
        self.enable_adascale = False
        self.logit_scaling = False
        ##
        self.num_classes = None
        self.feature_size = None
        self.return_combined_logits = False
        self.is_ood_eval = False

    def toggle_combined_logits(self):
        self.return_combined_logits = not (self.return_combined_logits)

    def feature_extractor(self, x):
        return self.get_features(x)

    @property
    def features(self):
        try:
            return self.model.features
        except Exception as e:
            print(
                "Warning: No features found in the model, using backbone features instead. Error: {}".format(
                    e
                )
            )
            return self.model.backbone.features

    @property
    def _input_size(self):
        return self.model._input_size

    def forward(
        self,
        x: torch.Tensor,
        return_feature=False,
        return_feature_list=False,
        threshold=None,
    ):
        features = self.get_features(x)
        scale = None
        if threshold is not None:
            features, scale = clip_features(self, features, threshold)
        logits = self.classifier(features)
        if scale is not None and self.logit_scaling:
            logits *= scale**2.0
        if return_feature:
            return logits, features
        if return_feature_list:
            # This wrapper exposes only the final representation, so retain
            # the list-based interface as a single-element sequence.
            return logits, [features]
        return logits

    def forward_threshold(self, x, threshold):
        logits = self(x, threshold=threshold)
        return logits

    def forward_rep(self, x):
        return self.get_features(x)

    def get_features(self, x):
        features = forward_rep(self.model, x)
        if self.feature_size is None:
            self.feature_size = features.view(x.shape[0], -1).size(1)
        return features

    def get_fc(self):
        """
        Returns the fully connected layer of the model.
        This is useful for strategies that require access to the classifier layer.
        """
        fc = self.get_fc_layer()
        return get_fc_w_b(fc, self.num_classes)

    def get_fc_layer(self):
        fc_layer = get_fc_layer(self.classifier, task_id=self.task_id)
        return fc_layer


class WrapModelMT(WrapModel, MultiTaskModule):
    def __init__(self, model, classifier):
        super().__init__(model, classifier)

    def forward(
        self,
        x: torch.Tensor,
        task_labels=None,
        return_feature=False,
        return_feature_list=False,
        threshold=None,
    ):
        features = self.get_features(x)
        scale = None
        if threshold is not None:
            features, scale = clip_features(self, features, threshold)
        if task_labels is None and self.task_id is not None:
            task_labels = self.task_id
        if scale is not None and self.logit_scaling:
            scale = scale**2.0
        else:
            scale = 1.0
        if task_labels is None:
            # When no task labels provided, use all heads and take max confidence
            res = {}
            for task_id in self.known_train_tasks_labels:
                res[task_id] = self.classifier(features, task_id) * scale
            logits = res
        else:
            logits = self.classifier(features, task_labels) * scale
        if self.return_combined_logits and isinstance(logits, dict):
            # combine all logits into a single tensor
            combined_logits = []
            for task_id in sorted(logits.keys()):
                combined_logits.append(logits[task_id])
            logits = torch.cat(combined_logits, dim=1)
        if return_feature:
            return logits, features
        if return_feature_list:
            # This wrapper exposes only the final representation, so retain
            # the list-based interface as a single-element sequence.
            return logits, [features]
        return logits
