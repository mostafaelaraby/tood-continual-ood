"""Focused regression tests for OODPostprocessorManager state handling.

Run with:
    python -m pytest -q Tests/test_ood_manager.py
"""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch.nn as nn

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

# ``docker/requirements.txt`` deliberately pins NumPy below 2.0 because the
# OpenOOD dependency ``imgaug`` still accesses ``np.sctypes``.  Skip this
# integration-level import test in an environment outside that contract;
# changing production code to emulate a removed third-party NumPy API would
# conceal a broken environment.
if int(np.__version__.split(".")[0]) >= 2:
    raise unittest.SkipTest(
        "OOD manager tests require the project's supported NumPy version (<2.0)."
    )

from managers.ood_manager import OODPostprocessorManager


class _FailingPostprocessor:
    def inference(self, *args, **kwargs):
        raise RuntimeError("intentional inference failure")


class TestOODManagerStateHandling(unittest.TestCase):
    def test_inference_resets_ood_mode_after_failure(self):
        """An exception must not leave the shared model in OOD mode."""
        model = nn.Linear(2, 2)
        manager = OODPostprocessorManager.__new__(OODPostprocessorManager)
        manager.cl_strategy = SimpleNamespace(model=model)
        manager.postprocessor = _FailingPostprocessor()

        with self.assertRaisesRegex(RuntimeError, "intentional"):
            manager.inference(data_loader=[])

        self.assertFalse(model.is_ood_eval)

    def test_feature_only_id_evaluation_uses_scenario_task_routing(self):
        """Feature-only output is returned correctly without strategy config."""
        model = nn.Linear(2, 2)
        manager = OODPostprocessorManager.__new__(OODPostprocessorManager)
        manager.config = SimpleNamespace(
            scenario=SimpleNamespace(return_task_id=True)
        )
        manager.cl_strategy = SimpleNamespace(model=model)
        manager.thresholds = {}
        manager.analyze_bias_vs_distance = lambda task_id: None

        feature_values = np.array([[1.0, 2.0]])
        calls = []

        def fake_inference(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return np.array([0]), np.array([0.9]), np.array([0]), feature_values
            return np.array([0]), np.array([0.8]), np.array([0])

        manager.inference = fake_inference
        result = manager.eval_for_id_task(
            id_train_loader=object(),
            id_test_loader=object(),
            current_task_id=1,
            id_data_task_id=0,
            return_feature=True,
        )

        _, _, id_conf, train_conf, _, id_features = result
        self.assertTrue(np.array_equal(id_conf, np.array([0.9])))
        self.assertTrue(np.array_equal(train_conf, np.array([0.8])))
        self.assertTrue(np.array_equal(id_features, feature_values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
