"""No-TensorFlow regression of settings and the actual optimizer factory dispatch."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from utils.optimizer_config import resolve_base_optimizer

ROOT = Path(__file__).resolve().parents[1]


class Adam:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class AdamW(Adam):
    pass


class LegacyAdamW(AdamW):
    pass


def factory(native=True, experimental=False):
    # Compile the actual function from train.py without importing its TF models.
    tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "build_optimizer")
    optimizers = SimpleNamespace(Adam=Adam)
    if native:
        optimizers.AdamW = AdamW
    if experimental:
        optimizers.experimental = SimpleNamespace(AdamW=AdamW)
    namespace = dict(Dict=dict, tf=SimpleNamespace(keras=SimpleNamespace(optimizers=optimizers)),
                     LegacyDecoupledAdamW=LegacyAdamW)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "train.py", "exec"), namespace)
    return namespace["build_optimizer"]


class OptimizerTest(unittest.TestCase):
    def test_defaults_preserve_adamw(self):
        self.assertEqual(resolve_base_optimizer({"weight_decay": .05}), ("adamw", .05))
        self.assertIs(type(factory()({"training": {"weight_decay": .05}}, .001)), AdamW)

    def test_actual_adam_dispatch(self):
        optimizer = factory()({"training": {"base_optimizer": "adam", "weight_decay": 0}}, .0003)
        self.assertIs(type(optimizer), Adam)
        self.assertNotIn("weight_decay", optimizer.kwargs)
        self.assertEqual(optimizer.kwargs["learning_rate"], .0003)

    def test_adamw_paths_and_lr(self):
        for native, experimental, cls in ((True, False, AdamW),
                                          (False, True, AdamW), (False, False, LegacyAdamW)):
            optimizer = factory(native, experimental)(
                {"training": {"base_optimizer": "adamw", "weight_decay": .05}}, .00001)
            self.assertIs(type(optimizer), cls)
            self.assertEqual(optimizer.kwargs["weight_decay"], .05)
            self.assertEqual(optimizer.kwargs["learning_rate"], .00001)

    def test_invalid_settings_rejected(self):
        for training in ({"base_optimizer": "sgd"}, {"weight_decay": -1},
                         {"weight_decay": float("nan")}, {"weight_decay": float("inf")},
                         {"base_optimizer": "adam", "weight_decay": .05}):
            with self.subTest(training=training), self.assertRaises(ValueError):
                factory()({"training": training}, .001)

    def test_pair_config_difference(self):
        from config import load_config
        prefix = "rafdb_siglip2_semantic_stable_v5_lgsa_vlm_kd"
        a = load_config(ROOT / ("config_" + prefix + "_adam_sam.yaml"))
        w = load_config(ROOT / ("config_" + prefix + "_adamw_sam.yaml"))
        self.assertEqual(resolve_base_optimizer(a["training"]), ("adam", 0.))
        self.assertEqual(resolve_base_optimizer(w["training"]), ("adamw", .05))
        for cfg in (a, w):
            cfg.pop("source")
            cfg["model"].pop("name")
            cfg["paths"].pop("output_dir")
            cfg["training"].pop("base_optimizer")
            cfg["training"].pop("weight_decay")
        self.assertEqual(a, w)


if __name__ == "__main__":
    unittest.main()
