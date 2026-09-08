#!/usr/bin/env python3
"""Select exactly ten V5 checkpoints by validation Macro F1, then test only the winner."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rafdb_pose_common import CLASSES, RAF_TO_MODEL, file_hash, resolve_image, write_csv

NAME = "rafdb_siglip2_semantic_stable_v5_combined_ultimate"
CHECKPOINTS = tuple(f"ckpt-{n}" for n in (12, 13, 15, 16, 17, 36, 39, 41, 42, 49))
COLUMNS = ("checkpoint", "val_accuracy", "val_macro_f1", "fear_f1", "disgust_f1")


def read_manifest(data_root, manifest, expected=0):
    """Read existing RAF image manifests without remapping the CSV on disk."""
    manifest = Path(manifest).resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        keys = reader.fieldnames or []
        path_key = next((k for k in ("pixels", "image_path", "filepath", "path", "image", "file")
                         if k in keys), None)
        label_key = next((k for k in ("emotion", "label", "target", "class", "y")
                          if k in keys), None)
        if not path_key or not label_key:
            raise ValueError(f"Missing image path or label column: {manifest}")
        rows = [dict(sample_id=i,
                     image_path=resolve_image(row[path_key], Path(data_root), manifest.parent),
                     label=int(row[label_key])) for i, row in enumerate(reader)]
    labels = {r["label"] for r in rows}
    if labels == set(range(1, 8)):
        for row in rows:
            row["label"] = RAF_TO_MODEL[row["label"]]
    elif labels != set(range(7)):
        raise ValueError(f"Expected all seven RAF classes, found {labels}: {manifest}")
    if expected and len(rows) != expected:
        raise ValueError(f"Unexpected sample count {len(rows)}, expected {expected}: {manifest}")
    if len({r["image_path"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate image paths: {manifest}")
    return rows


def checkpoint_fingerprint(prefix):
    prefix = Path(prefix)
    index = Path(str(prefix) + ".index")
    shards = sorted(prefix.parent.glob(prefix.name + ".data-*"))
    matches = [re.fullmatch(re.escape(prefix.name) + r"\.data-(\d+)-of-(\d+)", p.name)
               for p in shards]
    if not index.is_file() or not matches or any(m is None for m in matches):
        raise FileNotFoundError(f"Missing checkpoint components: {prefix}")
    total = int(matches[0][2])
    if (len(shards) != total or {int(m[1]) for m in matches} != set(range(total))
            or any(int(m[2]) != total for m in matches)):
        raise ValueError(f"Incomplete checkpoint shards: {prefix}")
    return {p.name[len(prefix.name):]: file_hash(p) for p in [index, *shards]}


def resolve_checkpoints(directories):
    """Duplicate names are allowed only for byte-identical saved checkpoints."""
    result = {}
    for name in CHECKPOINTS:
        matches = sorted({(Path(directory) / name).resolve() for directory in directories
                          if (Path(directory) / (name + ".index")).is_file()})
        if not matches:
            raise FileNotFoundError(f"Required {name} is missing from {directories}; refusing subset evaluation.")
        fingerprints = [checkpoint_fingerprint(p) for p in matches]
        if any(value != fingerprints[0] for value in fingerprints[1:]):
            raise ValueError(f"Ambiguous {name}: different checkpoint contents at {matches}. "
                             "Supply the exact source directories.")
        result[name] = dict(prefix=str(matches[0]), aliases=[str(p) for p in matches],
                            sha256=fingerprints[0])
    return result


def select_checkpoint(rows):
    if len(rows) != len(CHECKPOINTS) or {row["checkpoint"] for row in rows} != set(CHECKPOINTS):
        raise ValueError("Selection requires all ten distinct requested checkpoints.")
    for row in rows:
        for key in ("val_macro_f1", "val_accuracy"):
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"Invalid {key} for {row['checkpoint']}")
        if row["epoch"] < 0:
            raise ValueError("Checkpoint epoch must be read from the checkpoint.")
    return max(rows, key=lambda r: (r["val_macro_f1"], r["val_accuracy"], -r["epoch"]))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"config_{NAME}.yaml")
    parser.add_argument("--checkpoint-dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("hflip50", "no_tta"), default="hflip50",
                        help="Fixed BEFORE selection, reused on test. No weight sweep.")
    args = parser.parse_args()
    if any(k.startswith("MGR_") for k in os.environ):
        raise ValueError("Unset MGR_* overrides; evaluation must use the recorded config.")
    from config import load_config
    cfg = load_config(args.config)
    if cfg["model"]["name"] != NAME:
        raise ValueError("Use the V5 Combined Ultimate config belonging to these checkpoints.")
    assert cfg["data"]["num_classes"] == 7 and not cfg["data"].get("mask_dir")
    assert cfg["model"]["checkpoint_path"] is None
    assert not cfg["model"].get("use_multistage_adaptive_fusion", False)
    directories = args.checkpoint_dirs or [
        Path(cfg["paths"]["output_dir"]) / "checkpoints" / kind for kind in ("best", "best_loss")
    ]
    manifest = resolve_checkpoints(directories)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite or auto-resume a test evaluation.
    config_hash = file_hash(args.config)
    # Ordering must be stable for inspectable per-sample results.
    cfg = copy.deepcopy(cfg)
    cfg["runtime"]["tf_data_deterministic"] = True
    protocol = ("0.5*original_logits + 0.5*flip_logits" if args.protocol == "hflip50"
                else "original_logits")
    report = dict(config=str(args.config.resolve()), config_sha256=config_hash,
                  effective_config=cfg, checkpoints=manifest, class_names=CLASSES,
                  protocol=protocol, criterion="max val_macro_f1; tie: val_accuracy, earlier stored epoch",
                  candidate_ids=list(CHECKPOINTS), selection_split="val")
    write_json(out / "protocol.json", report)

    import tensorflow as tf
    from train import build_model, configure_tensorflow_runtime, configure_gpus, get_class_names
    from datasets.fer2013 import SplitRecords, make_dataset, _resolve_split_csv_dir
    from metrics.classification import classification_metrics

    configure_tensorflow_runtime(cfg)
    configure_gpus(cfg)
    tf.keras.utils.set_random_seed(cfg["seed"]["random_seed"])
    assert get_class_names(cfg) == CLASSES
    data_root = _resolve_split_csv_dir(Path(cfg["data"]["data_path"]))
    # Only validation is read/inferred before the selection decision is saved.
    def load_split(split):
        csv_path = data_root / f"{split}.csv"
        digest = file_hash(csv_path)
        records = read_manifest(data_root, csv_path, expected=3068 if split == "test" else 0)
        if {r["label"] for r in records} != set(range(7)):
            raise ValueError(f"{split}: expected all seven classes.")
        write_csv(out / f"{split}_samples.csv", records)
        rec = SplitRecords(np.array([r["image_path"] for r in records], dtype=object),
                           np.array([r["label"] for r in records], dtype=np.int64),
                           np.array([r["sample_id"] for r in records], dtype=np.int64), None)
        dataset = make_dataset(rec, cfg, split=split, training=False, replicas=1)
        return records, dataset, csv_path, digest

    val_records, val_dataset, val_csv, val_hash = load_split("val")
    model = build_model(cfg)
    first_inputs, _ = next(iter(val_dataset))
    model(first_inputs, training=False)
    epoch_variable = tf.Variable(-1, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(model=model, epoch=epoch_variable)

    def restore(name):
        item = manifest[name]
        prefix = Path(item["prefix"])
        if checkpoint_fingerprint(prefix) != item["sha256"]:
            raise ValueError(f"Checkpoint changed: {prefix}")
        stored_epoch = int(tf.train.load_variable(str(prefix), "epoch/.ATTRIBUTES/VARIABLE_VALUE"))
        status = checkpoint.read(str(prefix))
        status.assert_nontrivial_match()
        status.assert_existing_objects_matched()  # Complete model + epoch, not just some matching tensors.
        status.expect_partial()  # Saved optimizer state is deliberately unused.
        if int(epoch_variable.numpy()) != stored_epoch:
            raise ValueError("Restored epoch mismatch.")
        if checkpoint_fingerprint(prefix) != item["sha256"]:
            raise ValueError(f"Checkpoint changed during restore: {prefix}")
        print(f"CHECKPOINT_MODEL_RESTORE_OK checkpoint={name} stored_epoch={stored_epoch} "
              f"prefix={prefix}", flush=True)
        return stored_epoch

    @tf.function(reduce_retracing=True, jit_compile=False)
    def predict(inputs):
        original = tf.cast(model(inputs, training=False)["logits"], tf.float32)
        if args.protocol == "no_tta":
            return original
        flipped_inputs = dict(inputs)
        flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
        flipped = tf.cast(model(flipped_inputs, training=False)["logits"], tf.float32)
        return 0.5 * original + 0.5 * flipped

    def evaluate(name, split, records, dataset, csv_path, digest):
        expected_labels = np.array([r["label"] for r in records], dtype=np.int64)
        all_logits, offset = [], 0
        for inputs, labels in dataset:
            actual_labels = labels.numpy()
            np.testing.assert_array_equal(actual_labels,
                                          expected_labels[offset:offset + len(actual_labels)])
            logits = predict(inputs).numpy()
            if logits.shape != (len(actual_labels), 7) or not np.isfinite(logits).all():
                raise ValueError(f"Invalid logits: {split} {name}")
            all_logits.append(logits)
            offset += len(actual_labels)
        if offset != len(records):
            raise ValueError(f"Incomplete {split}: {offset}/{len(records)}")
        if file_hash(csv_path) != digest or file_hash(args.config) != config_hash:
            raise ValueError("Source config/manifest changed during evaluation.")
        if checkpoint_fingerprint(Path(manifest[name]["prefix"])) != manifest[name]["sha256"]:
            raise ValueError(f"Checkpoint changed during evaluation: {name}")
        logits = np.concatenate(all_logits)
        predictions = logits.argmax(axis=-1)
        result = classification_metrics(expected_labels, predictions, CLASSES)
        result.update(checkpoint=name, prefix=manifest[name]["prefix"],
                      split=split, protocol=protocol, samples=len(records),
                      manifest_sha256=digest)
        np.savez_compressed(out / f"{split}_{name}_logits.npz",
                            logits=logits, labels=expected_labels,
                            sample_ids=np.array([r["sample_id"] for r in records]))
        write_csv(out / f"{split}_{name}_predictions.csv",
                  [dict(r, prediction=int(p), predicted_class=CLASSES[int(p)])
                   for r, p in zip(records, predictions)])
        write_json(out / f"{split}_{name}_metrics.json", result)
        return result

    print(f"Target Split: VALIDATION | N={len(val_records)} | protocol={protocol}", flush=True)
    rows = []
    for name in CHECKPOINTS:
        epoch = restore(name)
        result = evaluate(name, "val", val_records, val_dataset, val_csv, val_hash)
        row = dict(checkpoint=name, val_accuracy=result["accuracy"], val_macro_f1=result["macro_f1"],
                   fear_f1=result["classification_report"]["fear"]["f1-score"],
                   disgust_f1=result["classification_report"]["disgust"]["f1-score"],
                   epoch=epoch, prefix=manifest[name]["prefix"])
        rows.append(row)
        write_csv(out / "validation_checkpoints.csv",
                  [{key: row[key] for key in COLUMNS} for row in rows], fields=COLUMNS)
        print("VAL " + json.dumps(row), flush=True)

    winner = select_checkpoint(rows)
    selection = dict(winner=winner, candidates=rows, protocol=protocol,
                     selection="validation_macro_f1", tie_break="validation_accuracy_then_earlier_stored_epoch",
                     validation_manifest_sha256=val_hash, test_evaluated=False)
    # Freeze and persist the decision BEFORE loading test records or evaluating test.
    write_json(out / "selection.json", selection)
    print(f"SELECTED_ON_VALIDATION {winner['checkpoint']} "
          f"macro_f1={winner['val_macro_f1']:.8f}", flush=True)

    restore(winner["checkpoint"])
    test_records, test_dataset, test_csv, test_hash = load_split("test")
    if {r["image_path"] for r in val_records} & {r["image_path"] for r in test_records}:
        raise ValueError("Validation/test image overlap.")
    write_json(out / "test_evaluation_started.json",
               dict(checkpoint=winner["checkpoint"], protocol=protocol, selection_file="selection.json"))
    test_result = evaluate(winner["checkpoint"], "test", test_records, test_dataset, test_csv, test_hash)
    write_json(out / "selected_test_metrics.json", test_result)
    write_json(out / "completed.json",
               dict(selected_checkpoint=winner["checkpoint"], test_evaluations=1,
                    validation_checkpoints=len(rows), protocol=protocol))
    print(f"SELECTED_TEST_ONCE_COMPLETE checkpoint={winner['checkpoint']} "
          f"accuracy={test_result['accuracy']:.8f} macro_f1={test_result['macro_f1']:.8f} "
          f"output={out}", flush=True)


if __name__ == "__main__":
    main()
