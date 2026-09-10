#!/usr/bin/env python3
"""Cache original/flip logits once per checkpoint, then sweep TTA for members and ensemble."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
from rafdb_pose_common import CLASSES, PROJECT_ROOT, read_test, file_hash, write_csv


def grid(step):
    if not np.isfinite(step) or step <= 0 or step > 1:
        raise ValueError('step must be in (0,1]')
    count = round(1 / step)
    if not np.isclose(count * step, 1):
        raise ValueError('step must divide 1 exactly, e.g. .05 or .01')
    return np.linspace(0, 1, count + 1)


def probabilities(original, flipped, weight):
    z = weight * original.astype(np.float64) + (1 - weight) * flipped.astype(np.float64)
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def metrics(labels, p):
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (labels, p.argmax(-1)), 1)
    n = int(cm.sum())
    support = cm.sum(1)
    denom = support + cm.sum(0)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(7), where=denom > 0)
    return dict(N=n, accuracy=float(cm.trace()/n), macro_f1=float(f1.mean()),
                weighted_f1=float(np.dot(f1, support)/n), confusion_matrix=cm.tolist(),
                per_class_accuracy={c: float(cm[i,i]/support[i]) if support[i] else None for i,c in enumerate(CLASSES)})


def sweep(original, flipped, labels, weights, names):
    result = {name: [] for name in [*names, 'ensemble']}
    for w in weights:
        # Important: softmax each member's mixed logits BEFORE averaging members.
        p = probabilities(original, flipped, float(w))
        for i, name in enumerate(names):
            result[name].append(dict(w_orig=float(w), w_flip=float(1-w), **metrics(labels, p[i])))
        result['ensemble'].append(dict(w_orig=float(w), w_flip=float(1-w), **metrics(labels, p.mean(0))))
    return result


def best_index(rows):
    # Fixed tie-break: accuracy, macro F1, nearest .5, smaller original weight.
    return max(range(len(rows)), key=lambda i: (rows[i]['accuracy'], rows[i]['macro_f1'],
                                              -abs(rows[i]['w_orig']-.5), -rows[i]['w_orig']))


def fingerprint(prefix):
    index = Path(str(prefix)+'.index')
    shards = sorted(prefix.parent.glob(prefix.name+'.data-*'))
    matches = [re.fullmatch(r'.*\.data-(\d+)-of-(\d+)', str(s)) for s in shards]
    if not index.is_file() or not matches or any(m is None for m in matches):
        raise ValueError(f'Missing checkpoint components: {prefix}')
    total = int(matches[0][2])
    if len(matches) != total or {int(m[1]) for m in matches} != set(range(total)) or any(int(m[2]) != total for m in matches):
        raise ValueError(f'Incomplete checkpoint shards: {prefix}')
    return {str(p): file_hash(p) for p in [index, *shards]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--step', type=float, default=.05)
    args = parser.parse_args()
    weights = grid(args.step)
    prefixes = sorted([p.with_suffix('') for p in args.checkpoint_dir.glob('ckpt-*.index')],
                      key=lambda p: int(p.name.split('-')[-1]))
    if len(prefixes) != 5:
        raise ValueError(f'Expected exactly 5 best_loss checkpoints, got {len(prefixes)}')
    fingerprints = {p.name: fingerprint(p) for p in prefixes}
    config_hash = file_hash(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(PROJECT_ROOT))
    import tensorflow as tf
    from config import load_config
    from train import build_model, configure_tensorflow_runtime, configure_gpus, get_class_names
    from datasets.fer2013 import SplitRecords, make_dataset, _resolve_split_csv_dir
    cfg = load_config(args.config)
    assert cfg['data']['num_classes'] == 7 and get_class_names(cfg) == CLASSES
    assert not cfg['data'].get('mask_dir'), 'This runner is for the mask-free RAF pipeline'
    assert tf.config.list_physical_devices('GPU'), 'GPU job requested but TensorFlow sees no GPU'
    cfg['runtime']['tf_data_deterministic'] = True
    configure_tensorflow_runtime(cfg)
    configure_gpus(cfg)
    tf.keras.utils.set_random_seed(cfg['seed']['random_seed'])
    root = _resolve_split_csv_dir(Path(cfg['data']['data_path']))
    records, datasets = {}, {}
    for split in ('val', 'test'):
        rows = read_test(root, root/f'{split}.csv', expected=3068 if split == 'test' else 0)
        records[split] = rows
        write_csv(args.output_dir/f'{split}_samples.csv', rows)
        rec = SplitRecords(np.array([r['image_path'] for r in rows], dtype=object),
                           np.array([r['label'] for r in rows]), np.arange(len(rows)), None)
        datasets[split] = make_dataset(rec, cfg, split=split, training=False, replicas=1)
    assert not ({r['image_path'] for r in records['val']} & {r['image_path'] for r in records['test']})
    first, _ = next(iter(datasets['val']))
    model = build_model(cfg)
    previous_ablation = getattr(model, 'ablation', None)
    previous_disable = getattr(model, 'disable_region_branch_when_cnn_only', None)
    try:
        if previous_ablation is not None:
            model.ablation = 'no_mask'
        if previous_disable is not None:
            model.disable_region_branch_when_cnn_only = False
        model(first, training=False)
    finally:
        if previous_ablation is not None:
            model.ablation = previous_ablation
        if previous_disable is not None:
            model.disable_region_branch_when_cnn_only = previous_disable
    model(first, training=False)
    checkpoint = tf.train.Checkpoint(model=model)
    report = dict(config=str(args.config), config_sha256=config_hash, effective_config=cfg,
                  class_names=CLASSES, checkpoint_files=fingerprints, step=args.step,
                  protocol='softmax(w*original_logits+(1-w)*flip_logits), then equal mean of 5 member probabilities',
                  tensorflow_version=tf.__version__, mixed_precision=tf.keras.mixed_precision.global_policy().name,
                  selection='validation accuracy; ties: macro-F1, closest to .5, smaller w_orig', sweeps={})
    names = [p.name for p in prefixes]
    selected = {}
    summary = []
    for split in ('val', 'test'):
        labels = np.array([r['label'] for r in records[split]], dtype=np.int64)
        originals, flips = [], []
        for prefix in prefixes:
            if fingerprint(prefix) != fingerprints[prefix.name]:
                raise ValueError('Checkpoint changed during sweep')
            status = checkpoint.read(str(prefix))
            status.assert_nontrivial_match()
            status.assert_existing_objects_matched()
            status.expect_partial()
            print(f'CHECKPOINT_MODEL_RESTORE_OK {prefix} split={split}', flush=True)
            orig, flip, offset = [], [], 0
            for inputs, y in datasets[split]:
                batch_y = y.numpy()
                np.testing.assert_array_equal(batch_y, labels[offset:offset+len(batch_y)])
                orig.append(tf.cast(model(inputs, training=False)['logits'], tf.float32).numpy())
                flipped = dict(inputs, image=tf.image.flip_left_right(inputs['image']))
                flip.append(tf.cast(model(flipped, training=False)['logits'], tf.float32).numpy())
                offset += len(batch_y)
            a, b = np.concatenate(orig), np.concatenate(flip)
            if a.shape != (len(labels), 7) or b.shape != a.shape or not np.isfinite([a,b]).all():
                raise ValueError('Invalid or incomplete logits')
            if fingerprint(prefix) != fingerprints[prefix.name]:
                raise ValueError('Checkpoint changed during inference')
            np.savez_compressed(args.output_dir/f'{split}_{prefix.name}_logits.npz',
                                original=a, flipped=b, labels=labels)
            originals.append(a)
            flips.append(b)
            print(f'LOGITS_CACHED {split} {prefix.name} N={offset}', flush=True)
        results = sweep(np.stack(originals), np.stack(flips), labels, weights, names)
        report['sweeps'][split] = results
        if split == 'val':
            selected = {name: best_index(rows) for name, rows in results.items()}
        for name, rows in results.items():
            for i, row in enumerate(rows):
                summary.append(dict(split=split, model=name, selected_on_val=i == selected[name],
                                    **{k:row[k] for k in ('w_orig','w_flip','N','accuracy','macro_f1','weighted_f1')}))
                print(f'{split.upper():4} {name:10} orig={row["w_orig"]:.2f} flip={row["w_flip"]:.2f} '
                      f'Acc={row["accuracy"]*100:.4f}% MacroF1={row["macro_f1"]:.4f}'
                      + (' VAL-SELECTED' if i == selected[name] else ''), flush=True)
    test = report['sweeps']['test']
    report['test_at_validation_selected_weights'] = {name: rows[selected[name]] for name,rows in test.items()}
    report['test_best_exploratory_NOT_unbiased'] = {name: rows[best_index(rows)] for name,rows in test.items()}
    report['test_no_tta'] = {name: rows[-1] for name,rows in test.items()}
    write_csv(args.output_dir/'sweep.csv', summary)
    if file_hash(args.config) != config_hash:
        raise ValueError('Config changed during evaluation')
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, default=str, allow_nan=False), encoding='utf-8')
    print('FINAL ENSEMBLE (validation-selected TTA):', report['test_at_validation_selected_weights']['ensemble'])
    print('TEST-BEST EXPLORATORY ONLY:', report['test_best_exploratory_NOT_unbiased']['ensemble'])
    print(f'SWEEP_COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
