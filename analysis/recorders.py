# Minimal reproduction surface: only CKA + AUROC-deterioration paths are retained.
import copy

import numpy as np
import pandas as pd
import torch

from utils import device
from utils.helpers import (
    ModelStateContext,
    forward_rep,
    get_embeddings_for_alignment,
    set_task_id,
)


class CKARecorder:
    """
    Computes Centered Kernel Alignment (CKA) between representations
    at different training stages to measure structural similarity of
    the feature space over continual learning tasks.

    Tracks:
    - Reference embeddings stored when each task is first learned
    - CKA similarity between reference and later representations (drift)
    - Adjacent-task CKA (T-1 vs T representation similarity)
    """

    def __init__(self, kernel="linear", max_samples=1000):
        """
        Args:
            kernel: 'linear' or 'rbf'.
            max_samples: Cap on samples used for CKA (memory/compute).
        """
        self.kernel = kernel
        self.max_samples = max_samples
        self.reference_embeddings = {}  # {data_task_id: np.ndarray (N, D)}
        self.ood_reference_embeddings = {}  # {ood_type: np.ndarray}
        self.ood_reference_task = (
            {}
        )  # {ood_type: int} – task when reference was stored
        self.cka_results = []  # list of dicts

    # ------ kernel helpers ------

    @staticmethod
    def _centering_matrix(n):
        return np.eye(n) - np.ones((n, n)) / n

    @staticmethod
    def _linear_kernel(X):
        return X @ X.T

    @staticmethod
    def _rbf_kernel(X, sigma=None):
        sq_norms = np.sum(X**2, axis=1, keepdims=True)
        sq_dists = sq_norms + sq_norms.T - 2.0 * X @ X.T
        sq_dists = np.maximum(sq_dists, 0.0)
        if sigma is None:
            mask = sq_dists > 0
            sigma = (
                np.sqrt(np.median(sq_dists[mask]) / 2.0) if mask.any() else 1.0
            )
        return np.exp(-sq_dists / (2.0 * sigma**2))

    def _kernel(self, X):
        if self.kernel == "rbf":
            return self._rbf_kernel(X)
        return self._linear_kernel(X)

    @staticmethod
    def _hsic(K, L, H):
        n = K.shape[0]
        return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)

    @staticmethod
    def _center_kernel(kernel):
        """Center a Gram matrix without materializing or multiplying by H.

        ``H @ K @ H`` is exactly equivalent to subtracting the row and
        column means and adding the grand mean.  The latter is O(n²), while
        the former performs cubic-time matrix multiplications.
        """
        row_mean = kernel.mean(axis=1, keepdims=True)
        col_mean = kernel.mean(axis=0, keepdims=True)
        return kernel - row_mean - col_mean + kernel.mean()

    def _compute_cka(self, X, Y):
        """
        CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
        """
        assert (
            X.shape[0] == Y.shape[0]
        ), "X and Y must have the same number of samples"
        n = X.shape[0]
        if n < 2:
            return 0.0

        # The previous implementation evaluated trace(K @ H @ L @ H)
        # directly, which requires several O(n³) matrix multiplications.
        # Centering each kernel first makes HSIC an element-wise inner
        # product.  Cast after kernel construction to retain the previous
        # float64 centering precision and metric values.
        K = self._center_kernel(
            self._kernel(X).astype(np.float64, copy=False)
        )
        L = self._center_kernel(
            self._kernel(Y).astype(np.float64, copy=False)
        )
        scale = float((n - 1) ** 2)
        hsic_xy = np.einsum("ij,ij->", K, L, optimize=True) / scale
        hsic_xx = np.einsum("ij,ij->", K, K, optimize=True) / scale
        hsic_yy = np.einsum("ij,ij->", L, L, optimize=True) / scale

        denom = np.sqrt(hsic_xx * hsic_yy)
        if denom < 1e-10:
            return 0.0
        return float(hsic_xy / denom)

    # ------ recording ------

    def _subsample(self, embeddings):
        # Deterministic first-N selection to ensure consistent pairing across
        # calls (CKA requires matched sample indices).
        if embeddings.shape[0] > self.max_samples:
            return embeddings[: self.max_samples]
        return embeddings

    def update(
        self, model, dataloader, data_task_id, current_task_id, max_batches=10
    ):
        """
        Extract embeddings for *data_task_id*'s data through the current model
        (at training stage *current_task_id*).

        * When data_task_id == current_task_id the embeddings are stored as
          the reference snapshot.
        * When data_task_id < current_task_id CKA is computed against the
          stored reference to quantify representation drift.
        """
        set_task_id(model, data_task_id)
        model.eval()

        with ModelStateContext(model):
            with torch.no_grad():
                embeddings, _, _ = get_embeddings_for_alignment(
                    dataloader,
                    lambda x: forward_rep(model, x),
                    device,
                    task_identifier=str(data_task_id),
                    dataset_name="ID",
                    num_batches=max_batches,
                    return_sids=True,
                )

        if embeddings is None or embeddings.shape[0] == 0:
            return

        embeddings = self._subsample(embeddings)

        # Store reference when the task is first learned
        if data_task_id == current_task_id:
            self.reference_embeddings[data_task_id] = copy.deepcopy(embeddings)

        # Compute CKA drift against stored reference
        if (
            data_task_id in self.reference_embeddings
            and data_task_id < current_task_id
        ):
            ref = self.reference_embeddings[data_task_id]
            n = min(len(embeddings), len(ref))
            cka = self._compute_cka(embeddings[:n], ref[:n])
            self.cka_results.append(
                {
                    "task": data_task_id,
                    "measured_at": current_task_id,
                    "cka_similarity": cka,
                    "comparison_type": "drift_from_origin",
                    "n_samples": n,
                }
            )
            print(
                f"CKA(Task {data_task_id} @ T{data_task_id} vs T{current_task_id}) = {cka:.4f}"
            )

        # Adjacent-task comparison (current vs previous task reference)
        prev = current_task_id - 1
        if (
            data_task_id == current_task_id
            and prev in self.reference_embeddings
        ):
            ref = self.reference_embeddings[prev]
            n = min(len(embeddings), len(ref))
            cka = self._compute_cka(embeddings[:n], ref[:n])
            self.cka_results.append(
                {
                    "task": prev,
                    "measured_at": current_task_id,
                    "cka_similarity": cka,
                    "comparison_type": "adjacent",
                    "n_samples": n,
                }
            )
            print(
                f"CKA(Task {prev}, Task {current_task_id}) [Adjacent] = {cka:.4f}"
            )

    def update_ood(
        self, model, dataloader, current_task_id, ood_type, max_batches=10
    ):
        """
        Track how OOD representations drift as the model trains on more tasks.

        On first encounter for a given *ood_type* the embeddings are stored as
        the reference. On subsequent calls, CKA is computed between current
        embeddings and the stored reference (same data, different model state).
        """
        set_task_id(model, current_task_id)
        model.eval()

        with ModelStateContext(model):
            with torch.no_grad():
                embeddings, _, _ = get_embeddings_for_alignment(
                    dataloader,
                    lambda x: forward_rep(model, x),
                    device,
                    task_identifier=str(current_task_id),
                    dataset_name=ood_type,
                    num_batches=max_batches,
                    return_sids=True,
                )

        if embeddings is None or embeddings.shape[0] == 0:
            return

        embeddings = self._subsample(embeddings)

        # Store reference on first encounter
        if ood_type not in self.ood_reference_embeddings:
            self.ood_reference_embeddings[ood_type] = copy.deepcopy(embeddings)
            self.ood_reference_task[ood_type] = current_task_id
            print(
                f"CKA: Stored {ood_type} reference at T{current_task_id} "
                f"({len(embeddings)} samples)"
            )
            return

        # Compute CKA drift against stored OOD reference
        ref = self.ood_reference_embeddings[ood_type]
        n = min(len(embeddings), len(ref))
        cka = self._compute_cka(embeddings[:n], ref[:n])
        ref_task = self.ood_reference_task[ood_type]
        self.cka_results.append(
            {
                "task": ood_type,
                "measured_at": current_task_id,
                "cka_similarity": cka,
                "comparison_type": "ood_drift",
                "n_samples": n,
            }
        )
        print(
            f"CKA({ood_type} @ T{ref_task} vs T{current_task_id}) = {cka:.4f}"
        )

    def finalize(self):
        if not self.cka_results:
            return pd.DataFrame(
                columns=[
                    "task",
                    "measured_at",
                    "cka_similarity",
                    "comparison_type",
                    "n_samples",
                ]
            )
        return pd.DataFrame(self.cka_results)
