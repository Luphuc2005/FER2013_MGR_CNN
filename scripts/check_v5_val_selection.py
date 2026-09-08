#!/usr/bin/env python3
"""Offline regression checks for selection logic and checkpoint discovery; no TensorFlow."""
from __future__ import annotations
import ast
import copy
from pathlib import Path
import tempfile
import unittest

from select_v5_checkpoint_val_macro_f1 import (
    CHECKPOINTS, COLUMNS, checkpoint_fingerprint, resolve_checkpoints, select_checkpoint, read_manifest,
)


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(checkpoint=name, epoch=int(name.split("-")[1]),
                     val_accuracy=0.9, val_macro_f1=0.8, fear_f1=0.6, disgust_f1=0.7)
                for name in CHECKPOINTS]

    def test_macro_f1_is_primary_and_test_metrics_ignored(self):
        rows = self.rows()
        rows[1].update(val_macro_f1=0.81, val_accuracy=0.85, test_accuracy=0.1)
        rows[2].update(val_accuracy=0.99, test_accuracy=1.0)
        self.assertEqual(select_checkpoint(rows)["checkpoint"], "ckpt-13")

    def test_ties_accuracy_then_stored_epoch_not_filename(self):
        rows = self.rows()
        rows[3]["val_accuracy"] = 0.92
        rows[4].update(val_accuracy=0.92, epoch=1)
        self.assertEqual(select_checkpoint(rows)["checkpoint"], "ckpt-17")

    def test_reject_missing_duplicate_and_nan(self):
        rows = self.rows()
        for invalid in (rows[:-1], rows[:-1] + [copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError):
                select_checkpoint(invalid)
        rows[0]["val_macro_f1"] = float("nan")
        with self.assertRaises(ValueError):
            select_checkpoint(rows)

    def test_components_and_ambiguous_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, b = root / "best", root / "best_loss"
            a.mkdir()
            b.mkdir()
            for name in CHECKPOINTS:
                (a / (name + ".index")).write_bytes(b"synthetic-index")
                (a / (name + ".data-00000-of-00001")).write_bytes(b"synthetic-data")
            name = CHECKPOINTS[0]
            for extension in (".index", ".data-00000-of-00001"):
                (b / (name + extension)).write_bytes((a / (name + extension)).read_bytes())
            self.assertEqual(len(resolve_checkpoints([a, b])), 10)
            self.assertEqual(len(resolve_checkpoints([a, b])[name]["aliases"]), 2)
            (b / (name + ".data-00000-of-00001")).write_bytes(b"other")
            with self.assertRaises(ValueError):
                resolve_checkpoints([a, b])
            (a / (name + ".data-00000-of-00001")).unlink()
            with self.assertRaises(FileNotFoundError):
                checkpoint_fingerprint(a / name)
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoints([root / "missing"])

    def test_manifest_headers_mapping_order_and_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(7):
                (root / f"{i}.jpg").write_bytes(b"fixture")
            manifest = root / "val.csv"
            for header in ("pixels", "image_path"):
                for raw in (False, True):
                    body = f"{header},label\n" + "".join(
                        f"{i}.jpg,{i + int(raw)}\n" for i in reversed(range(7)))
                    manifest.write_text(body, encoding="utf-8")
                    before = manifest.read_bytes()
                    rows = read_manifest(root, manifest, expected=7)
                    expected = [6, 0, 4, 3, 1, 2, 5] if raw else list(reversed(range(7)))
                    self.assertEqual([r["label"] for r in rows], expected)
                    self.assertEqual([Path(r["image_path"]).name for r in rows],
                                     [f"{i}.jpg" for i in reversed(range(7))])
                    self.assertEqual([r["sample_id"] for r in rows], list(range(7)))
                    self.assertEqual(manifest.read_bytes(), before)
                    with self.assertRaises(ValueError):
                        read_manifest(root, manifest, expected=8)

    def test_required_csv_columns(self):
        self.assertEqual(COLUMNS, ("checkpoint", "val_accuracy", "val_macro_f1", "fear_f1", "disgust_f1"))

    def test_single_test_evaluation_after_selection_is_persisted(self):
        path = Path(__file__).with_name("select_v5_checkpoint_val_macro_f1.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
        tests = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "evaluate"
                 and len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "test"]
        self.assertEqual(len(tests), 1)
        selection_writes = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "write_json"
                            and n.args and "selection.json" in ast.unparse(n.args[0])]
        self.assertEqual(len(selection_writes), 1)
        self.assertLess(selection_writes[0].lineno, tests[0].lineno)
        for loop in [n for n in ast.walk(main) if isinstance(n, (ast.For, ast.While))]:
            self.assertNotIn(tests[0], list(ast.walk(loop)))


if __name__ == "__main__":
    unittest.main()
