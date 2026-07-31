"""
Unit tests for StrategyStateHelper in utils/helpers.py.

Run with:  python test_strategy_state_helper.py
"""

import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

# Make the test runnable from the repository root with either unittest or
# pytest.  Adding ``Tests/`` itself cannot resolve the sibling ``utils``
# package during pytest collection.
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)
from utils.helpers import StrategyStateHelper

# ---------------------------------------------------------------------------
# Minimal stubs (no Avalanche required)
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self, counter=0):
        self.train_exp_counter = counter


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)


class _ModelWithLambda(nn.Module):
    """Simulates ResidualBlock that embeds an unpicklable local lambda."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        # local lambda — cannot be pickled by standard pickle
        self._act = lambda x: x * 2

    def forward(self, x):
        return self._act(self.fc(x))


class _EvalClassifier:
    def __init__(self):
        self.class_means_dict = {"0": [1.0, 2.0], "1": [3.0, 4.0]}

    def replace_class_means_dict(self, d):
        self.class_means_dict = d


class _ModelWithNCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.eval_classifier = _EvalClassifier()


class _Strategy:
    """Minimal strategy stub."""

    def __init__(self, model=None, plugins=None, clock=None):
        self.model = model or _Model()
        self.plugins = plugins or []
        self.clock = clock or _Clock()
        # a few plain attrs that should be saved/restored
        self.some_counter = 42
        self.some_flag = True
        self._unpicklable = object()  # must be silently skipped


# ---------------------------------------------------------------------------
# Plugin stubs
# ---------------------------------------------------------------------------


class _SimplePlugin:
    """Plugin with only plain savable attributes."""

    def __init__(self):
        self.alpha = 0.5
        self.count = 7
        self.history = [1.0, 2.0, 3.0]
        self._skip_me = object()  # not savable

    def after_training_exp(self, strategy, **kwargs):
        pass


class _PluginWithModule:
    """Plugin where an nn.Module attribute is *always* initialised."""

    def __init__(self):
        self.lr = 0.01
        self.bias_layer = nn.Linear(4, 4)

    def after_training_exp(self, strategy, **kwargs):
        pass


class _BiasLayerProxy(nn.Module):
    """
    Minimal stand-in for Avalanche's BiasLayer.
    Constructor signature: BiasLayerProxy(clss) where clss is a LongTensor of
    class indices — mirrors BiasLayer(targets.uniques).
    The 'clss' buffer is stored in the state_dict, so _reconstruct_module can
    call cls(state_dict['clss']) to rebuild it without knowing the class count.
    """

    def __init__(self, clss=None):
        super().__init__()
        if clss is None:
            clss = torch.tensor([], dtype=torch.long)
        if not isinstance(clss, torch.Tensor):
            clss = torch.tensor(list(clss), dtype=torch.long)
        self.register_buffer("clss", clss)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(len(clss)))


class _BicLikePlugin:
    """
    Plugin that mirrors BiC: bias_layer starts as None and is created lazily
    inside after_training_exp (only after the first experience).
    """

    def __init__(self):
        self.lr = 0.001
        self.bias_layer = None  # created lazily

    def after_training_exp(self, strategy, **kwargs):
        # Lazily create the module (as BiC does after exp 0 via bias_correction_step)
        if self.bias_layer is None:
            clss = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            self.bias_layer = _BiasLayerProxy(clss)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_load(strategy_src, strategy_dst, device="cpu"):
    """Round-trip save→load using a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        StrategyStateHelper.save(strategy_src, path)
        result = StrategyStateHelper.load(strategy_dst, path, device=device)
    finally:
        os.unlink(path)
        lock = path + ".lock"
        if os.path.exists(lock):
            os.unlink(lock)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStrategyAttrs(unittest.TestCase):
    """Strategy-level primitive attributes are saved and restored."""

    def test_primitive_attrs_round_trip(self):
        src = _Strategy()
        src.some_counter = 99
        src.some_flag = False

        dst = _Strategy()
        dst.some_counter = 0
        dst.some_flag = True

        ok = _save_load(src, dst)
        self.assertTrue(ok)
        self.assertEqual(dst.some_counter, 99)
        self.assertFalse(dst.some_flag)

    def test_clock_restored(self):
        src = _Strategy(clock=_Clock(counter=5))
        dst = _Strategy(clock=_Clock(counter=0))

        _save_load(src, dst)
        self.assertEqual(dst.clock.train_exp_counter, 5)

    def test_model_weights_restored(self):
        src = _Strategy()
        # Set known weights
        nn.init.constant_(src.model.fc.weight, 7.0)

        dst = _Strategy()
        nn.init.constant_(dst.model.fc.weight, 0.0)

        _save_load(src, dst)
        self.assertTrue(
            torch.allclose(
                dst.model.fc.weight, torch.full_like(dst.model.fc.weight, 7.0)
            )
        )


class TestPluginPlainAttrs(unittest.TestCase):
    """Plugin plain (non-module) attributes are saved and restored."""

    def test_plain_attrs_round_trip(self):
        plugin_src = _SimplePlugin()
        plugin_src.alpha = 0.99
        plugin_src.count = 42
        plugin_src.history = [10.0, 20.0]

        plugin_dst = _SimplePlugin()

        src = _Strategy(plugins=[plugin_src])
        dst = _Strategy(plugins=[plugin_dst])
        _save_load(src, dst)

        self.assertAlmostEqual(plugin_dst.alpha, 0.99)
        self.assertEqual(plugin_dst.count, 42)
        self.assertEqual(plugin_dst.history, [10.0, 20.0])

    def test_unsavable_attrs_skipped_silently(self):
        plugin_src = _SimplePlugin()
        src = _Strategy(plugins=[plugin_src])
        dst = _Strategy(plugins=[_SimplePlugin()])
        # Should not raise even though plugin._skip_me is an object()
        _save_load(src, dst)


class TestPluginModuleInitialised(unittest.TestCase):
    """nn.Module plugin attributes that exist at load time are state_dict-restored."""

    def test_module_weights_restored(self):
        plugin_src = _PluginWithModule()
        nn.init.constant_(plugin_src.bias_layer.weight, 3.0)

        plugin_dst = _PluginWithModule()
        nn.init.constant_(plugin_dst.bias_layer.weight, 0.0)

        src = _Strategy(plugins=[plugin_src])
        dst = _Strategy(plugins=[plugin_dst])
        _save_load(src, dst)

        self.assertTrue(
            torch.allclose(
                plugin_dst.bias_layer.weight,
                torch.full_like(plugin_dst.bias_layer.weight, 3.0),
            )
        )

    def test_plain_and_module_attrs_together(self):
        plugin_src = _PluginWithModule()
        plugin_src.lr = 0.123
        nn.init.constant_(plugin_src.bias_layer.weight, 5.0)

        plugin_dst = _PluginWithModule()
        plugin_dst.lr = 0.0

        src = _Strategy(plugins=[plugin_src])
        dst = _Strategy(plugins=[plugin_dst])
        _save_load(src, dst)

        self.assertAlmostEqual(plugin_dst.lr, 0.123)
        self.assertTrue(
            torch.allclose(
                plugin_dst.bias_layer.weight,
                torch.full_like(plugin_dst.bias_layer.weight, 5.0),
            )
        )


class TestBicLikeImmediateReconstruction(unittest.TestCase):
    """
    When bias_layer is None at load time, _reconstruct_module must rebuild it
    immediately inside load() — no deferred hook needed.

    This matters because training is SKIPPED when a checkpoint exists, so a
    hook on after_training_exp would never fire.  By task N+1, bias_layer must
    already be non-None or distillation asserts will crash.
    """

    def _round_trip(self, plugin_src, plugin_dst):
        src = _Strategy(plugins=[plugin_src])
        dst = _Strategy(plugins=[plugin_dst])
        _save_load(src, dst)

    def _make_bias_layer(self, alpha_val, beta_val):
        clss = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        layer = _BiasLayerProxy(clss)
        nn.init.constant_(layer.alpha, alpha_val)
        nn.init.constant_(layer.beta, beta_val)
        return layer

    def test_bias_layer_reconstructed_immediately(self):
        """bias_layer is None at load time but must be non-None after load()."""
        plugin_src = _BicLikePlugin()
        plugin_src.bias_layer = self._make_bias_layer(9.0, -1.0)

        plugin_dst = _BicLikePlugin()  # bias_layer starts as None
        self.assertIsNone(plugin_dst.bias_layer)

        self._round_trip(plugin_src, plugin_dst)

        # Must be restored WITHOUT calling after_training_exp
        self.assertIsInstance(plugin_dst.bias_layer, nn.Module)

    def test_bias_layer_weights_correct_after_reconstruction(self):
        plugin_src = _BicLikePlugin()
        plugin_src.bias_layer = self._make_bias_layer(9.0, -1.0)

        plugin_dst = _BicLikePlugin()
        self._round_trip(plugin_src, plugin_dst)

        self.assertTrue(
            torch.allclose(
                plugin_dst.bias_layer.alpha,
                torch.full_like(plugin_dst.bias_layer.alpha, 9.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                plugin_dst.bias_layer.beta,
                torch.full_like(plugin_dst.bias_layer.beta, -1.0),
            )
        )

    def test_no_deferred_hook_installed(self):
        """Reconstruction is immediate — no after_training_exp patching needed."""
        plugin_src = _BicLikePlugin()
        plugin_src.bias_layer = self._make_bias_layer(1.0, 0.0)

        plugin_dst = _BicLikePlugin()
        self._round_trip(plugin_src, plugin_dst)

        self.assertFalse(getattr(plugin_dst, "_deferred_hook_installed", False))
        self.assertNotIn("after_training_exp", plugin_dst.__dict__)

    def test_skip_training_scenario(self):
        """
        Simulate the real failing scenario: tasks 0-2 checkpoints are loaded
        (training skipped), then task 3 training starts.  bias_layer must be
        non-None before task 3's distillation assert fires.
        """
        plugin_src = _BicLikePlugin()
        clss = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        plugin_src.bias_layer = _BiasLayerProxy(clss)
        nn.init.constant_(plugin_src.bias_layer.alpha, 5.0)

        # Simulate loading tasks 1 and 2 checkpoints into the same plugin
        # (training skipped each time, after_training_exp never called)
        plugin_live = _BicLikePlugin()
        strategy = _Strategy(plugins=[plugin_live])

        for _ in range(2):
            src_strategy = _Strategy(plugins=[plugin_src])
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                path = f.name
            try:
                StrategyStateHelper.save(src_strategy, path)
                StrategyStateHelper.load(strategy, path)
            finally:
                os.unlink(path)
                lock = path + ".lock"
                if os.path.exists(lock):
                    os.unlink(lock)

        # bias_layer must be available before task 3 training (no after_training_exp called)
        self.assertIsInstance(plugin_live.bias_layer, nn.Module)


class TestPickleSafety(unittest.TestCase):
    """Saving must not crash even when the model contains unpicklable lambdas."""

    def test_model_with_lambda_saves_without_error(self):
        src = _Strategy(model=_ModelWithLambda())
        dst = _Strategy(model=_ModelWithLambda())
        # Should not raise pickle errors
        _save_load(src, dst)

    def test_model_weights_still_restored_with_lambda_model(self):
        src = _Strategy(model=_ModelWithLambda())
        nn.init.constant_(src.model.fc.weight, 4.0)

        dst = _Strategy(model=_ModelWithLambda())
        nn.init.constant_(dst.model.fc.weight, 0.0)

        _save_load(src, dst)
        self.assertTrue(
            torch.allclose(
                dst.model.fc.weight, torch.full_like(dst.model.fc.weight, 4.0)
            )
        )


class TestNCMClassMeans(unittest.TestCase):
    """Special-case NCM class means are saved and restored."""

    def test_ncm_means_round_trip(self):
        model = _ModelWithNCM()
        model.eval_classifier.class_means_dict = {"0": [9.0], "1": [8.0]}

        src = _Strategy(model=model)
        dst = _Strategy(model=_ModelWithNCM())

        _save_load(src, dst)
        self.assertEqual(
            dst.model.eval_classifier.class_means_dict, {"0": [9.0], "1": [8.0]}
        )


class TestLegacyFormat(unittest.TestCase):
    """Old flat-dict checkpoint format is still loadable."""

    def test_legacy_flat_dict_loaded(self):
        plugin_dst = _SimplePlugin()
        plugin_dst.alpha = 0.0

        dst = _Strategy(plugins=[plugin_dst])

        # Manually craft a legacy checkpoint (flat dict, no _attrs/_modules keys)
        legacy_checkpoint = {
            "model": dst.model.state_dict(),
            "steps": 3,
            "strategy_attrs": {},
            "plugins": {"_SimplePlugin": {"alpha": 0.77, "count": 11}},
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(legacy_checkpoint, path)
            StrategyStateHelper.load(dst, path)
        finally:
            os.unlink(path)
            lock = path + ".lock"
            if os.path.exists(lock):
                os.unlink(lock)

        self.assertAlmostEqual(plugin_dst.alpha, 0.77)
        self.assertEqual(plugin_dst.count, 11)


class TestLoadReturnValue(unittest.TestCase):
    """load() returns False for missing file, True on success."""

    def test_returns_false_for_missing_file(self):
        dst = _Strategy()
        result = StrategyStateHelper.load(dst, "/nonexistent/path/ckpt.pt")
        self.assertFalse(result)

    def test_returns_true_on_success(self):
        src = _Strategy()
        dst = _Strategy()
        result = _save_load(src, dst)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
