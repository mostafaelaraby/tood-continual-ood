import csv
import os
from typing import Counter, Dict, List

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


def select_top_k_labels(labels, k=10):
    """
    Select the k most frequent labels from a list of predictions.

    Args:
        labels (list): List of predicted labels
        k (int): Number of top labels to select

    Returns:
        tuple: (top_k_labels, frequencies) - Lists of labels and their frequencies
    """
    # Count frequencies of each label
    label_counts = Counter(labels)

    # Get top k labels and their counts
    top_k = label_counts.most_common(k)

    # Separate labels and frequencies
    top_labels, frequencies = zip(*top_k)

    return list(top_labels), list(frequencies)


def split_dataloader(
    net, dataloader, target_classes=[0], train_size=0.8, batch_size=None, stratify=True
):
    """
    Split a PyTorch DataLoader into two DataLoaders with optional stratified split,
    filtering for specific target classes.

    Args:
        dataloader: Original DataLoader
        target_classes: List of class labels to include in the split (default: [1])
        train_size: Proportion of data for first split (default: 0.8)
        batch_size: Batch size for new DataLoaders (default: original batch size)
        stratify: Whether to perform stratified split (default: True)

    Returns:
        first_loader, second_loader: Two DataLoaders containing the split filtered data
    """
    # Get the original dataset
    try:
        dataloader.shuffle = False
    except:
        print(
            "failed to disable shuffling set to {} of dataloader".format(
                dataloader.shuffle
            )
        )
    dataset = dataloader.dataset

    # Use original batch size if none specified
    if batch_size is None:
        batch_size = dataloader.batch_size

    labels = []
    target_indices = []
    for _, batch in enumerate(dataloader):
        preds = net(batch["data"].cuda()).argmax(dim=1).cpu()
        labels.extend([pred.item() for pred in preds])
    # remap target classes to the actual class labels
    if target_classes is not None:
        target_classes, _ = select_top_k_labels(labels, k=len(target_classes))
    for idx, batch in enumerate(dataset):
        if idx >= len(labels):
            break
        if target_classes is None or labels[idx] in target_classes:
            target_indices.append(idx)

    # Create a filtered dataset
    filtered_dataset = Subset(dataset, target_indices)
    filtered_labels = [labels[i] for i in target_indices]

    if len(filtered_dataset) == 0:
        raise ValueError(f"No samples found for target classes {target_classes}")

    if stratify:
        # Create stratified split indices for filtered dataset
        split = StratifiedShuffleSplit(
            n_splits=1, train_size=train_size, random_state=42
        )

        # Get indices for both splits
        indices1, indices2 = next(
            split.split(X=range(len(filtered_dataset)), y=filtered_labels)
        )
    else:
        # Simple random split if no stratification
        indices = range(len(filtered_dataset))
        split_idx = int(len(filtered_dataset) * train_size)
        indices1, indices2 = indices[:split_idx], indices[split_idx:]

    # Create subset datasets from filtered dataset
    dataset1 = Subset(filtered_dataset, indices1)
    dataset2 = Subset(filtered_dataset, indices2)

    # Create new dataloaders
    loader1 = DataLoader(
        dataset1,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
    )

    loader2 = DataLoader(
        dataset2,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
    )

    return loader1, loader2


class OODEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(OODEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None

    def eval_ood(
        self,
        net: nn.Module,
        id_data_loaders: Dict[str, DataLoader],
        ood_data_loaders: Dict[str, Dict[str, DataLoader]],
        postprocessor: BasePostprocessor,
        fsood: bool = False,
    ):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert "test" in id_data_loaders, "id_data_loaders should have the key: test!"
        dataset_name = self.config.dataset.name

        if self.config.postprocessor.APS_mode:
            assert "val" in id_data_loaders
            assert "val" in ood_data_loaders
            self.hyperparam_search(
                net, id_data_loaders["val"], ood_data_loaders["val"], postprocessor
            )

        print(f"Performing inference on {dataset_name} dataset...", flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(net, id_data_loaders["test"])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        if fsood:
            # load csid data and compute confidence
            for dataset_name, csid_dl in ood_data_loaders["csid"].items():
                print(f"Performing inference on {dataset_name} dataset...", flush=True)
                csid_pred, csid_conf, csid_gt = postprocessor.inference(net, csid_dl)
                if self.config.recorder.save_scores:
                    self._save_scores(csid_pred, csid_conf, csid_gt, dataset_name)
                id_pred = np.concatenate([id_pred, csid_pred])
                id_conf = np.concatenate([id_conf, csid_conf])
                id_gt = np.concatenate([id_gt, csid_gt])

        # load nearood data and compute ood metrics
        print("\u2500" * 70, flush=True)
        self._eval_ood(
            net,
            [id_pred, id_conf, id_gt],
            ood_data_loaders,
            postprocessor,
            ood_split="nearood",
            id_data_loaders=id_data_loaders,
        )

        # load farood data and compute ood metrics
        print("\u2500" * 70, flush=True)
        self._eval_ood(
            net,
            [id_pred, id_conf, id_gt],
            ood_data_loaders,
            postprocessor,
            ood_split="farood",
            id_data_loaders=id_data_loaders,
        )

    def _eval_ood(
        self,
        net: nn.Module,
        id_list: List[np.ndarray],
        ood_data_loaders: Dict[str, Dict[str, DataLoader]],
        postprocessor: BasePostprocessor,
        ood_split: str = "nearood",
        id_data_loaders=None,
    ):
        print(f"Processing {ood_split}...", flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f"Performing inference on {dataset_name} dataset...", flush=True)
            if (
                hasattr(postprocessor, "update_ood_set")
                and postprocessor.use_oracle
                and postprocessor.synthetic_name == "local"
            ):
                ood_dl, val_ood = split_dataloader(
                    net,
                    ood_dl,
                    target_classes=None,
                    train_size=0.6,
                    batch_size=ood_dl.batch_size,
                    stratify=False,
                )
                postprocessor.update_ood_set(val_ood, dataset_name)
                id_pred, id_conf, id_gt = postprocessor.inference(
                    net, id_data_loaders["test"]
                )

            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            print(f"Computing metrics on {dataset_name} dataset...")

            ood_metrics = compute_all_metrics(conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)

        print("Computing mean metrics...", flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)

    def eval_ood_val(
        self,
        net: nn.Module,
        id_data_loaders: Dict[str, DataLoader],
        ood_data_loaders: Dict[str, DataLoader],
        postprocessor: BasePostprocessor,
    ):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert "val" in id_data_loaders
        assert "val" in ood_data_loaders
        if self.config.postprocessor.APS_mode:
            val_auroc = self.hyperparam_search(
                net, id_data_loaders["val"], ood_data_loaders["val"], postprocessor
            )
        else:
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loaders["val"]
            )
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loaders["val"]
            )
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            val_auroc = ood_metrics[1]
        return {"auroc": 100 * val_auroc}

    def _save_csv(self, metrics, dataset_name):
        [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics

        write_content = {
            "dataset": dataset_name,
            "FPR@95": "{:.2f}".format(100 * fpr),
            "AUROC": "{:.2f}".format(100 * auroc),
            "AUPR_IN": "{:.2f}".format(100 * aupr_in),
            "AUPR_OUT": "{:.2f}".format(100 * aupr_out),
            "ACC": "{:.2f}".format(100 * accuracy),
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print(
            "FPR@95: {:.2f}, AUROC: {:.2f}".format(100 * fpr, 100 * auroc),
            end=" ",
            flush=True,
        )
        print(
            "AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}".format(100 * aupr_in, 100 * aupr_out),
            flush=True,
        )
        print("ACC: {:.2f}".format(accuracy * 100), flush=True)
        print("\u2500" * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, "ood.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, "scores")
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name), pred=pred, conf=conf, label=gt)

    def eval_acc(
        self,
        net: nn.Module,
        data_loader: DataLoader,
        postprocessor: BasePostprocessor = None,
        epoch_idx: int = -1,
        fsood: bool = False,
        csid_data_loaders: DataLoader = None,
    ):
        """Returns the accuracy score of the labels and predictions.

        :return: float
        """
        if type(net) is dict:
            net["backbone"].eval()
        else:
            net.eval()
        self.id_pred, self.id_conf, self.id_gt = postprocessor.inference(
            net, data_loader
        )

        if fsood:
            assert csid_data_loaders is not None
            for dataset_name, csid_dl in csid_data_loaders.items():
                csid_pred, csid_conf, csid_gt = postprocessor.inference(net, csid_dl)
                self.id_pred = np.concatenate([self.id_pred, csid_pred])
                self.id_conf = np.concatenate([self.id_conf, csid_conf])
                self.id_gt = np.concatenate([self.id_gt, csid_gt])

        metrics = {}
        metrics["acc"] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics["epoch_idx"] = epoch_idx
        return metrics

    def report(self, test_metrics):
        print("Completed!", flush=True)

    def hyperparam_search(
        self,
        net: nn.Module,
        id_data_loader,
        ood_data_loader,
        postprocessor: BasePostprocessor,
    ):
        print("Starting automatic parameter search...")
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0
        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])
        hyperparam_combination = self.recursive_generator(hyperparam_list, count)
        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)
            id_pred, id_conf, id_gt = postprocessor.inference(net, id_data_loader)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_data_loader)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics[1]
            print("Hyperparam:{}, auroc:{}".format(hyperparam, aps_dict[index]))
            if ood_metrics[1] > max_auroc:
                max_auroc = ood_metrics[1]
        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])
        print("Final hyperparam: {}".format(postprocessor.get_hyperparam()))
        return max_auroc

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
