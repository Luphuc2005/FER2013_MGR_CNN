#!/usr/bin/env python3
"""Evaluate current RAF checkpoints on estimated pose groups, using existing preprocessing."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from rafdb_pose_common import CLASSES, GROUPS, PROJECT_ROOT, file_hash, pose_group, write_csv


def summarize(labels, probs, rows, mask, full_class_accuracy):
    indices = np.flatnonzero(mask)
    y = labels[indices]
    pred = probs[indices].argmax(axis=1)
    cm = np.zeros((7, 7), dtype=int)
    np.add.at(cm, (y, pred), 1)
    counts = cm.sum(axis=1)
    recall = np.divide(cm.diagonal(), counts, out=np.zeros(7), where=counts > 0)
    f1 = np.divide(2*cm.diagonal(), counts+cm.sum(axis=0), out=np.zeros(7), where=(counts+cm.sum(axis=0)) > 0)
    n = len(indices)
    return dict(N=n, detection_rate=float(np.mean([int(rows[i]['face_detected']) for i in indices])) if n else None,
                pose_valid_rate=float(np.mean([int(rows[i]['pose_valid']) for i in indices])) if n else None,
                accuracy=float(cm.trace()/n) if n else None, macro_f1=float(f1.mean()) if n else None,
                macro_f1_definition='mean over all seven classes; zero for undefined class F1',
                per_class_accuracy={c: float(recall[i]) if counts[i] else None for i, c in enumerate(CLASSES)},
                per_class_detection_rate={c: float(np.mean([int(rows[j]['face_detected']) for j in indices if labels[j] == i]))
                                          if counts[i] else None for i, c in enumerate(CLASSES)},
                per_class_pose_valid_rate={c: float(np.mean([int(rows[j]['pose_valid']) for j in indices if labels[j] == i]))
                                           if counts[i] else None for i, c in enumerate(CLASSES)},
                class_counts={c: int(counts[i]) for i, c in enumerate(CLASSES)},
                class_proportions={c: float(counts[i]/n) if n else None for i, c in enumerate(CLASSES)},
                composition_only_reference_accuracy=float(np.dot(counts, full_class_accuracy)/n) if n else None,
                confusion_matrix=cm.tolist())


def save_evaluation(out, tag, mode, rows, probs):
    out = out/tag/mode
    out.mkdir(parents=True, exist_ok=True)
    labels = np.array([int(r['label']) for r in rows])
    groups = np.array([r['subset'] for r in rows])
    pred = probs.argmax(1)
    full_class_acc = np.array([np.mean(pred[labels == i] == i) if np.any(labels == i) else 0 for i in range(7)])
    masks = {'full_test': np.ones(len(rows), dtype=bool), **{g: groups == g for g in GROUPS},
             'ge30_combined': np.isin(groups, ['30_to_lt45', 'ge45']),
             'detector_positive': np.array([int(r['face_detected']) == 1 for r in rows])}
    metrics = {g: summarize(labels, probs, rows, mask, full_class_acc) for g, mask in masks.items()}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    pred_rows = [dict(image_path=r['image_path'], label=int(r['label']), subset=r['subset'],
                      prediction=int(pred[i]), confidence=float(probs[i, pred[i]]),
                      **{f'p_{c}': float(probs[i, k]) for k, c in enumerate(CLASSES)}) for i, r in enumerate(rows)]
    write_csv(out/'predictions.csv', pred_rows)
    for group, values in metrics.items():
        with (out/f'confusion_{group}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['true/pred', *CLASSES])
            writer.writerows([c, *line] for c, line in zip(CLASSES, values['confusion_matrix']))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = np.array(values['confusion_matrix'])
        ax.imshow(cm, cmap='Blues')
        ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted', ylabel='True', title=f'{tag} {mode} {group} N={values["N"]}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
        fig.tight_layout()
        fig.savefig(out/f'confusion_{group}.png', dpi=140)
        plt.close(fig)
        acc = f'{values["accuracy"]:.4f}' if values['N'] else 'NA'
        print(f'{tag} {mode} {group}: N={values["N"]} accuracy={acc} macro_f1={values["macro_f1"]}', flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'config_rafdb_convnext_base_ms1m_adaptive_siglip2_confusion_v2.yaml')
    p.add_argument('--checkpoint-dir', type=Path, default=PROJECT_ROOT/'outputs/papers/rafdb_adaptive_siglip2_confusion_v2_v4/checkpoints/best')
    p.add_argument('--checkpoint', default='ckpt-35', help='Primary checkpoint chosen before test evaluation.')
    p.add_argument('--all-best', action='store_true', help='Also evaluate every checkpoint in best, and their equal-probability ensemble.')
    p.add_argument('--pose-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--no-tta', action='store_true', help='Only original-image inference. Default additionally reports config hflip TTA.')
    p.add_argument('--smoke-only', action='store_true', help='Check all input rows and checkpoint restores; inference only first batch, no metrics.')
    args = p.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    sys.path.insert(0, str(PROJECT_ROOT))
    import tensorflow as tf
    from config import load_config
    from train import build_model, configure_tensorflow_runtime, get_class_names
    from datasets.fer2013 import SplitRecords, make_dataset

    cfg = load_config(args.config)
    if get_class_names(cfg) != CLASSES or cfg['data']['num_classes'] != 7:
        raise ValueError('Expected RAF seven-class model in project class order.')
    if cfg['data'].get('mask_dir'):
        raise ValueError('This evaluator targets the current mask-free RAF SigLIP2 config; refusing to drop configured masks.')
    # Only evaluation runtime settings change in memory. No training/YAML writes.
    cfg['runtime'].update(use_mixed_precision=False, tf_data_deterministic=True,
                          batch_size_per_gpu=args.batch_size, tf_data_num_parallel_calls=8,
                          tf_data_private_threadpool_size=8, prefetch_buffer=2,
                          intra_op_threads=12, inter_op_threads=2)
    configure_tensorflow_runtime(cfg)
    tf.keras.mixed_precision.set_global_policy('float32')
    gpus = tf.config.list_physical_devices('GPU')
    if not args.cpu and not gpus:
        raise RuntimeError('GPU requested but TensorFlow sees none; use --cpu explicitly for CPU inference.')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(int(cfg['seed']['random_seed']))
    device = '/CPU:0' if args.cpu else '/GPU:0'

    pose_csv = args.pose_dir/'rafdb_test_pose.csv'
    with pose_csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    metadata = json.loads((args.pose_dir/'pose_metadata.json').read_text(encoding='utf-8'))
    if len(rows) != metadata['N'] or not rows or len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Pose manifest count or duplicate-path mismatch.')
    for r in rows:
        if not Path(r['image_path']).is_file() or int(r['label']) not in range(7) or r['subset'] not in GROUPS:
            raise ValueError(f'Invalid manifest row: {r}')
        if int(r['pose_valid']):
            angle = max(abs(float(r['yaw'])), abs(float(r['pitch'])))
            if not np.isclose(angle, float(r['pose_angle'])) or r['subset'] != pose_group(angle):
                raise ValueError('Pose angle/subset mismatch in CSV.')
        elif r['subset'] != 'undetected':
            raise ValueError('Invalid pose must belong to undetected.')
    labels = np.array([int(r['label']) for r in rows], dtype=np.int64)
    records = SplitRecords(np.array([r['image_path'] for r in rows], dtype=object), labels,
                           np.arange(len(rows), dtype=np.int64), None)
    dataset = make_dataset(records, cfg, split='test', training=False, replicas=1)
    first, first_labels = next(iter(dataset))
    print(f'POSE_EVAL_INPUT_OK N={len(rows)} shape={first["image"].shape} labels={labels.min()}..{labels.max()} device={device}', flush=True)

    primary = args.checkpoint_dir/args.checkpoint.removesuffix('.index')
    paths = [primary]
    if args.all_best:
        others = [q.with_suffix('') for q in args.checkpoint_dir.glob('ckpt-*.index')]
        paths += sorted([q for q in others if q != primary], key=lambda q: int(q.name.split('-')[-1]))
    fingerprints = {}
    for path in paths:
        if not re.fullmatch(r'ckpt-\d+', path.name) or not Path(str(path)+'.index').is_file():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        shards = sorted(path.parent.glob(path.name+'.data-*'))
        shard_ids = [re.fullmatch(r'.*\.data-(\d+)-of-(\d+)', str(s)) for s in shards]
        if not shards or any(m is None for m in shard_ids):
            raise ValueError(f'Missing/bad checkpoint shards: {path}')
        total = int(shard_ids[0][2])
        if len(shards) != total or {int(m[1]) for m in shard_ids} != set(range(total)) or any(int(m[2]) != total for m in shard_ids):
            raise ValueError(f'Incomplete checkpoint shards: {path}')
        fingerprints[path.name] = {str(q): {'size': q.stat().st_size, 'mtime_ns': q.stat().st_mtime_ns}
                                   for q in [Path(str(path)+'.index'), *shards]}
        fingerprints[path.name]['index_sha256'] = file_hash(Path(str(path)+'.index'))

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Use a fresh evaluation output: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tf.device(device):
        model = build_model(cfg)
        # Match train.py's eager build of conditional branches before strict restore.
        original_ablation = getattr(model, 'ablation', None)
        original_disable = getattr(model, 'disable_region_branch_when_cnn_only', None)
        try:
            if original_ablation is not None:
                model.ablation = 'no_mask'
            if original_disable is not None:
                model.disable_region_branch_when_cnn_only = False
            model(first, training=False)
        finally:
            if original_ablation is not None:
                model.ablation = original_ablation
            if original_disable is not None:
                model.disable_region_branch_when_cnn_only = original_disable
        model(first, training=False)
        checkpoint = tf.train.Checkpoint(model=model)

    use_tta = not args.no_tta and bool(cfg.get('tta', {}).get('enabled')) and bool(cfg.get('tta', {}).get('hflip'))
    orig_w = float(cfg.get('tta', {}).get('original_weight', .5))
    flip_w = float(cfg.get('tta', {}).get('flip_weight', .5))
    if not np.isfinite([orig_w, flip_w]).all() or min(orig_w, flip_w) < 0 or orig_w+flip_w <= 0:
        raise ValueError('Invalid TTA weights.')
    orig_w, flip_w = orig_w/(orig_w+flip_w), flip_w/(orig_w+flip_w)
    modes = ['no_tta'] + (['hflip_tta'] if use_tta else [])
    sums = {mode: np.zeros((len(rows), 7), dtype=np.float64) for mode in modes}
    report = dict(primary_checkpoint=str(primary), checkpoint_selection='user supplied primary; all-best members fixed before inference',
                  checkpoint_files=fingerprints, config_path=str(args.config), config_sha256=file_hash(args.config),
                  effective_config=cfg, pose_csv_sha256=file_hash(pose_csv), pose_metadata=metadata,
                  class_order=CLASSES, device=device, tensorflow_version=tf.__version__,
                  tta_weights=dict(original=orig_w, hflip=flip_w), ensemble='equal mean of member softmax probabilities', results={})
    for path in paths:
        for name, recorded in fingerprints[path.name].items():
            if name == 'index_sha256':
                continue
            stat = Path(name).stat()
            if (stat.st_size, stat.st_mtime_ns) != (recorded['size'], recorded['mtime_ns']):
                raise RuntimeError(f'Checkpoint changed after enumeration: {name}')
        with tf.device(device):
            # read() avoids adding a save_counter absent from Checkpoint.write checkpoints.
            status = checkpoint.read(str(path))
            status.assert_nontrivial_match()
            status.assert_existing_objects_matched()
            status.expect_partial()  # optimizer/epoch extras in training checkpoints intentionally unused
            print(f'CHECKPOINT_MODEL_RESTORE_OK {path} model_variables={len(model.variables)}', flush=True)
            arrays = {mode: [] for mode in modes}
            offset = 0
            for batch_index, (inputs, batch_labels) in enumerate(dataset):
                actual = batch_labels.numpy()
                if not np.array_equal(actual, labels[offset:offset+len(actual)]):
                    raise ValueError('Sample/label order drift between manifest and inference.')
                logits = tf.cast(model(inputs, training=False)['logits'], tf.float32)
                arrays['no_tta'].append(tf.nn.softmax(logits).numpy())
                if use_tta:
                    flipped = dict(inputs, image=tf.image.flip_left_right(inputs['image']))
                    other = tf.cast(model(flipped, training=False)['logits'], tf.float32)
                    arrays['hflip_tta'].append(tf.nn.softmax(orig_w*logits + flip_w*other).numpy())
                offset += len(actual)
                if batch_index % 20 == 0:
                    print(f'INFERENCE_PROGRESS {path.name} {offset}/{len(rows)}', flush=True)
                if args.smoke_only:
                    break
        for name, recorded in fingerprints[path.name].items():
            if name != 'index_sha256':
                stat = Path(name).stat()
                if (stat.st_size, stat.st_mtime_ns) != (recorded['size'], recorded['mtime_ns']):
                    raise RuntimeError(f'Checkpoint changed during evaluation: {name}')
        for mode in modes:
            probs = np.concatenate(arrays[mode])
            if not np.isfinite(probs).all() or probs.shape[1] != 7 or not np.allclose(probs.sum(1), 1, atol=1e-5):
                raise ValueError('Invalid model probabilities.')
            if args.smoke_only:
                continue
            if len(probs) != len(rows):
                raise ValueError('Incomplete full-test predictions.')
            sums[mode] += probs
            report['results'][f'{path.name}/{mode}'] = save_evaluation(args.output_dir, path.name, mode, rows, probs)
    if args.smoke_only:
        print('POSE_EVAL_SMOKE_OK (first batch only; no accuracy metrics produced)', flush=True)
        return
    if args.all_best:
        for mode in modes:
            report['results'][f'ensemble/{mode}'] = save_evaluation(args.output_dir, 'ensemble', mode, rows, sums[mode]/len(paths))
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, default=str, allow_nan=False), encoding='utf-8')
    summary = []
    interpretation = [
        'Estimated pose robustness on current RAF crops; not the official Wang Pose-RAF benchmark.',
        'N, detector success, valid-pose rate, class mix and absent classes must accompany accuracy.',
        'Per-class accuracy is recall within the true class. No sample is removed from full_test.',
        'composition_only_reference_accuracy uses FULL per-class accuracies weighted by each subset class mix; diagnostic only.',
        'inspect_*.jpg and inspect_samples.csv contain fixed-seed random examples including failed detections.',
        'Images have NOT been manually inspected by this script; review these sheets before reporting robustness.',
        'Higher subset accuracy does not establish that pose improves accuracy. Detection may select easier faces.',
        'Small subsets have unstable accuracy. Report N; do not select checkpoints/TTA on these test scores.',
    ]
    for key, groups in report['results'].items():
        for group, metric in groups.items():
            summary.append(dict(model_protocol=key, group=group, **{k: metric[k] for k in ['N', 'detection_rate', 'pose_valid_rate', 'accuracy', 'macro_f1']}))
            if group in ['30_to_lt45', 'ge45', 'ge30_combined'] and metric['N'] and metric['accuracy'] > groups['full_test']['accuracy']:
                interpretation.append(f'{key} {group}: accuracy above full_test, N={metric["N"]}; inspect class_counts, composition reference and detection failures. This is compatible with stable performance in the selected estimated-pose subset, not a causal benefit of pose.')
    write_csv(args.output_dir/'summary.csv', summary)
    (args.output_dir/'interpretation.txt').write_text('\n'.join(interpretation)+'\n', encoding='utf-8')
    print(f'POSE_ROBUSTNESS_EVAL_OK report={args.output_dir/"report.json"}', flush=True)


if __name__ == '__main__':
    main()
