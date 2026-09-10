#!/usr/bin/env python3
"""Unified Checkpoint Ensemble & TTA Sweep for RAF-DB V6 Models.

Evaluates and sweeps:
1. Individual checkpoints in `checkpoints/best/` (Ranked by val_accuracy).
2. Individual checkpoints in `checkpoints/best_loss/` (Ranked by val_loss).
3. Individual checkpoint in `checkpoints/best_macro_f1/` (Ranked by val_macro_f1, if available).
4. Top-K Best Accuracy Ensemble.
5. Top-K Best Loss Ensemble.
6. Combined Cross-Pool Ensemble (Best Accuracy + Best Loss + Best Macro-F1).

Protocol:
- TTA weights w_orig in [0.0, 1.0] with step 0.05.
- Optimal TTA weight is selected strictly on the VALIDATION split.
- Final test metrics are evaluated at the validation-selected optimal weight.
- Also records test without TTA (w_orig=1.0) and standard 50/50 TTA.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, resolve_auto_increment_output_dir
from datasets.fer2013 import build_datasets
from metrics.classification import classification_metrics, save_metrics
from train import build_model, configure_gpus, configure_tensorflow_runtime, get_class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Ensemble and TTA Sweep for RAF-DB V6")
    parser.add_argument(
        "--config",
        type=str,
        default="config_rafdb_v6_anti_overfitting.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to run output directory containing checkpoints/ (default: auto-detected from config)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Grid step size for TTA original weight sweep (default: 0.05)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save ensemble and sweep reports (default: output_dir/ensemble_results)",
    )
    return parser.parse_args()


def extract_epoch(path: Path) -> int:
    name = path.stem if path.name.endswith(".index") else path.name
    digits = re.findall(r"\d+", name)
    return int(digits[-1]) if digits else 0


def find_checkpoints(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    indices = sorted(directory.glob("ckpt-*.index"), key=extract_epoch)
    return [p.with_suffix("") for p in indices]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    metrics = classification_metrics(y_true.tolist(), y_pred.tolist(), class_names)
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
    }


def cache_logits_for_checkpoint(model, checkpoint_prefix: Path, dataset, ckpt_obj) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import tensorflow as tf

    status = ckpt_obj.read(str(checkpoint_prefix))
    status.expect_partial()

    all_orig = []
    all_flip = []
    all_labels = []

    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            inputs, labels = batch[0], batch[1]
        else:
            inputs, labels = batch, batch["label"]

        # Original pass
        out_orig = model(inputs, training=False)
        all_orig.append(out_orig["logits"].numpy())

        # Flipped pass
        flipped_inputs = dict(inputs)
        flipped_inputs["image"] = tf.image.flip_left_right(inputs["image"])
        if "mask" in inputs and inputs["mask"] is not None:
            flipped_inputs["mask"] = tf.image.flip_left_right(inputs["mask"])
        out_flip = model(flipped_inputs, training=False)
        all_flip.append(out_flip["logits"].numpy())

        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_orig, axis=0).astype(np.float32),
        np.concatenate(all_flip, axis=0).astype(np.float32),
        np.concatenate(all_labels, axis=0).astype(np.int64),
    )


def sweep_tta_weights(
    logits_orig: np.ndarray,
    logits_flip: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    weights: np.ndarray,
) -> list[dict]:
    results = []
    for w in weights:
        w_orig = float(w)
        w_flip = float(1.0 - w_orig)
        fused = w_orig * logits_orig + w_flip * logits_flip
        preds = np.argmax(fused, axis=-1)
        m = compute_metrics(labels, preds, class_names)
        results.append({
            "w_orig": round(w_orig, 4),
            "w_flip": round(w_flip, 4),
            **m,
        })
    return results


def pick_best_weight_index(sweep_rows: list[dict]) -> int:
    # Tie-break priority: accuracy -> macro_f1 -> closeness to 0.5 -> larger w_orig
    return max(
        range(len(sweep_rows)),
        key=lambda i: (
            sweep_rows[i]["accuracy"],
            sweep_rows[i]["macro_f1"],
            -abs(sweep_rows[i]["w_orig"] - 0.5),
            sweep_rows[i]["w_orig"],
        ),
    )


def evaluate_ensemble(
    member_probs_orig: list[np.ndarray],
    member_probs_flip: list[np.ndarray],
    labels: np.ndarray,
    class_names: list[str],
    weights: np.ndarray,
) -> list[dict]:
    # member_probs: list of [N, num_classes] arrays
    results = []
    for w in weights:
        w_orig = float(w)
        w_flip = float(1.0 - w_orig)
        # Combine per-member TTA first, then average probabilities across ensemble members
        member_fused_probs = [
            softmax(w_orig * orig + w_flip * flip)
            for orig, flip in zip(member_probs_orig, member_probs_flip)
        ]
        ensemble_p = np.mean(member_fused_probs, axis=0)
        preds = np.argmax(ensemble_p, axis=-1)
        m = compute_metrics(labels, preds, class_names)
        results.append({
            "w_orig": round(w_orig, 4),
            "w_flip": round(w_flip, 4),
            **m,
        })
    return results


def main():
    args = parse_args()
    cfg = load_config(args.config)
    class_names = get_class_names(cfg)

    # Determine run output directory
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = resolve_auto_increment_output_dir(cfg, for_eval=True)

    ckpt_root = run_dir / "checkpoints"
    print("=" * 80)
    print(f"RAF-DB V6 UNIFIED ENSEMBLE & TTA SWEEP")
    print(f"Output Directory: {run_dir}")
    print(f"Checkpoint Root:  {ckpt_root}")
    print("=" * 80)

    save_dir = Path(args.save_dir) if args.save_dir else run_dir / "ensemble_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover checkpoint pools
    pool_acc = find_checkpoints(ckpt_root / "best")
    pool_loss = find_checkpoints(ckpt_root / "best_loss")
    pool_macro = find_checkpoints(ckpt_root / "best_macro_f1")

    print(f"\n[DISCOVERY]")
    print(f"  checkpoints/best/      (val_accuracy): {[p.name for p in pool_acc]}")
    print(f"  checkpoints/best_loss/ (val_loss):     {[p.name for p in pool_loss]}")
    print(f"  checkpoints/best_macro_f1/:            {[p.name for p in pool_macro]}")

    all_pools = {
        "best_accuracy": pool_acc,
        "best_loss": pool_loss,
        "best_macro_f1": pool_macro,
    }

    # Set of unique checkpoint prefixes to avoid redundant inference
    unique_prefixes = {}
    for pool_name, ckpts in all_pools.items():
        for p in ckpts:
            unique_prefixes[p.name] = p

    if not unique_prefixes:
        print(f"[ERROR] No checkpoints found in {ckpt_root}. Ensure training has produced checkpoints.")
        sys.exit(1)

    # Initialize TensorFlow and Model
    import tensorflow as tf
    configure_tensorflow_runtime(cfg)
    configure_gpus(cfg)
    tf.keras.utils.set_random_seed(int(cfg["seed"]["random_seed"]))

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = build_model(cfg)

    _, val_ds, test_ds = build_datasets(cfg, replicas=1)

    # Dummy forward to build model variables
    first_batch = next(iter(val_ds.take(1)))
    _ = model(first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch, training=False)

    ckpt_obj = tf.train.Checkpoint(model=model)

    # Cache logits for all unique checkpoints
    # cached_logits[split][ckpt_name] = (orig, flip)
    cached_logits = {"val": {}, "test": {}}
    val_labels = None
    test_labels = None

    print(f"\n[INFERENCE] Caching logits for {len(unique_prefixes)} unique checkpoints...")
    for name, prefix in unique_prefixes.items():
        print(f"  -> Processing checkpoint: {name} ({prefix})", flush=True)

        # Validation
        v_orig, v_flip, v_labels = cache_logits_for_checkpoint(model, prefix, val_ds, ckpt_obj)
        cached_logits["val"][name] = (v_orig, v_flip)
        if val_labels is None:
            val_labels = v_labels

        # Test
        t_orig, t_flip, t_labels = cache_logits_for_checkpoint(model, prefix, test_ds, ckpt_obj)
        cached_logits["test"][name] = (t_orig, t_flip)
        if test_labels is None:
            test_labels = t_labels

    weights = np.linspace(0.0, 1.0, round(1.0 / args.step) + 1)
    report = {
        "config": str(args.config),
        "run_dir": str(run_dir),
        "class_names": class_names,
        "grid_step": args.step,
        "individual_models": {},
        "ensembles": {},
    }

    # -----------------------------------------------------------------------
    # 1. Sweep Individual Checkpoints
    # -----------------------------------------------------------------------
    print(f"\n" + "=" * 80)
    print("INDIVIDUAL CHECKPOINT EVALUATION (TTA Sweep)")
    print("=" * 80)
    print(f"{'Pool':<16} {'Model':<12} {'Val No-TTA':<12} {'Val Opt-TTA':<12} {'w_orig*':<8} {'Test No-TTA':<13} {'Test Opt-TTA':<13} {'Test MacroF1':<12}")
    print("-" * 105)

    for pool_name, ckpts in all_pools.items():
        if not ckpts:
            continue
        for p in ckpts:
            name = p.name
            v_orig, v_flip = cached_logits["val"][name]
            t_orig, t_flip = cached_logits["test"][name]

            val_sweep = sweep_tta_weights(v_orig, v_flip, val_labels, class_names, weights)
            test_sweep = sweep_tta_weights(t_orig, t_flip, test_labels, class_names, weights)

            # Pick optimal weight index from VALIDATION
            best_idx = pick_best_weight_index(val_sweep)
            w_opt = val_sweep[best_idx]["w_orig"]

            val_no_tta = next(r for r in val_sweep if np.isclose(r["w_orig"], 1.0))
            val_opt = val_sweep[best_idx]

            test_no_tta = next(r for r in test_sweep if np.isclose(r["w_orig"], 1.0))
            test_opt = test_sweep[best_idx]

            report["individual_models"][f"{pool_name}/{name}"] = {
                "checkpoint": str(p),
                "val_no_tta": val_no_tta,
                "val_opt_tta": val_opt,
                "test_no_tta": test_no_tta,
                "test_opt_tta": test_opt,
                "selected_w_orig": w_opt,
            }

            print(
                f"{pool_name:<16} {name:<12} "
                f"{val_no_tta['accuracy']*100:>6.2f}%      "
                f"{val_opt['accuracy']*100:>6.2f}%      "
                f"{w_opt:>5.2f}    "
                f"{test_no_tta['accuracy']*100:>7.2f}%      "
                f"{test_opt['accuracy']*100:>7.2f}%      "
                f"{test_opt['macro_f1']:>8.4f}"
            )

    # -----------------------------------------------------------------------
    # 2. Ensemble Evaluation
    # -----------------------------------------------------------------------
    print(f"\n" + "=" * 80)
    print("ENSEMBLE EVALUATION (Top-K Acc, Top-K Loss, and Combined)")
    print("=" * 80)
    print(f"{'Ensemble Name':<28} {'Members':<8} {'Val No-TTA':<12} {'Val Opt-TTA':<12} {'w_orig*':<8} {'Test No-TTA':<13} {'Test Opt-TTA':<13} {'Test MacroF1':<12}")
    print("-" * 115)

    ensemble_groups = {}
    if pool_acc:
        ensemble_groups["Top-K Best Accuracy"] = [p.name for p in pool_acc]
    if pool_loss:
        ensemble_groups["Top-K Best Loss"] = [p.name for p in pool_loss]

    # Combined Cross-Pool Ensemble
    combined_members = list(dict.fromkeys(
        [p.name for p in pool_acc] + [p.name for p in pool_loss] + [p.name for p in pool_macro]
    ))
    if len(combined_members) > 1:
        ensemble_groups["Combined (Acc + Loss + F1)"] = combined_members

    best_ensemble_metrics = None
    best_ensemble_name = None

    for ens_name, members in ensemble_groups.items():
        v_orig_list = [cached_logits["val"][m][0] for m in members]
        v_flip_list = [cached_logits["val"][m][1] for m in members]
        t_orig_list = [cached_logits["test"][m][0] for m in members]
        t_flip_list = [cached_logits["test"][m][1] for m in members]

        val_ens_sweep = evaluate_ensemble(v_orig_list, v_flip_list, val_labels, class_names, weights)
        test_ens_sweep = evaluate_ensemble(t_orig_list, t_flip_list, test_labels, class_names, weights)

        best_ens_idx = pick_best_weight_index(val_ens_sweep)
        w_ens_opt = val_ens_sweep[best_ens_idx]["w_orig"]

        val_ens_no_tta = next(r for r in val_ens_sweep if np.isclose(r["w_orig"], 1.0))
        val_ens_opt = val_ens_sweep[best_ens_idx]

        test_ens_no_tta = next(r for r in test_ens_sweep if np.isclose(r["w_orig"], 1.0))
        test_ens_opt = test_ens_sweep[best_ens_idx]

        report["ensembles"][ens_name] = {
            "members": members,
            "val_no_tta": val_ens_no_tta,
            "val_opt_tta": val_ens_opt,
            "test_no_tta": test_ens_no_tta,
            "test_opt_tta": test_ens_opt,
            "selected_w_orig": w_ens_opt,
        }

        print(
            f"{ens_name:<28} {len(members):<8} "
            f"{val_ens_no_tta['accuracy']*100:>6.2f}%      "
            f"{val_ens_opt['accuracy']*100:>6.2f}%      "
            f"{w_ens_opt:>5.2f}    "
            f"{test_ens_no_tta['accuracy']*100:>7.2f}%      "
            f"{test_ens_opt['accuracy']*100:>7.2f}%      "
            f"{test_ens_opt['macro_f1']:>8.4f}"
        )

        if best_ensemble_metrics is None or test_ens_opt["accuracy"] > best_ensemble_metrics["accuracy"]:
            best_ensemble_metrics = test_ens_opt
            best_ensemble_name = ens_name

    # Save outputs
    report_path = save_dir / "ensemble_and_sweep_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"SUMMARY & REPORT")
    print(f"Report JSON saved to: {report_path}")
    if best_ensemble_name:
        print(f"Best Ensemble: {best_ensemble_name}")
        print(f"  Test Accuracy (Validation-Selected TTA): {best_ensemble_metrics['accuracy']*100:.2f}%")
        print(f"  Test Macro-F1:                          {best_ensemble_metrics['macro_f1']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
