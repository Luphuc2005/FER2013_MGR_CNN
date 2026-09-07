#!/usr/bin/env python3
"""Static contract by default; --smoke requires real RAF CSVs, TF and GPU."""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config
from utils.semantic_schedule import resolve_lambda_sem


def contract(path):
    cfg = load_config(path)
    baseline = load_config(ROOT/'config_rafdb_siglip2_semantic_stable_v3.yaml')
    model = cfg['model']
    assert model['use_soft_regional_pooling'] is True
    assert model['semantic_projector_dropout'] == .10
    assert model['hard_margin'] == model['clip_semantic']['hard_margin'] == .15
    assert model['classifier_dropout1'] == .40
    assert model['use_adaptive_granularity'] and model['ablation'] == 'adaptive_clip_confusion'
    assert model['semantic_logit_scale'] == model['clip_semantic']['semantic_logit_scale'] == 20
    assert cfg['training']['visual_extractor_lr'] == 1e-5
    assert cfg['training']['optimizer'] == 'sam' and cfg['training']['base_optimizer'] == 'adamw'
    assert cfg['training']['resume'] is False and model['checkpoint_path'] is None
    assert cfg['data']['image_size'] == 112
    assert model['name'] == 'rafdb_siglip2_semantic_stable_v4_combined'
    assert Path(cfg['paths']['output_dir']) == ROOT/'outputs/papers/rafdb_siglip2_semantic_stable_v4_combined'
    same = copy.deepcopy(cfg)
    same['model'].pop('use_soft_regional_pooling')
    same['model'].pop('semantic_projector_dropout')
    same['model']['name'] = baseline['model']['name']
    same['model']['hard_margin'] = baseline['model']['hard_margin']
    same['model']['clip_semantic']['hard_margin'] = baseline['model']['clip_semantic']['hard_margin']
    same['source'] = baseline['source']
    same['paths']['output_dir'] = baseline['paths']['output_dir']
    assert same == baseline, 'Unexpected change beyond three intended improvements + experiment identity'
    for epoch, value in {1:.1, 4:.1, 5:.1, 6:.11, 9:.14, 10:.15, 11:.15, 60:.15}.items():
        assert abs(resolve_lambda_sem(cfg, epoch)-value) < 1e-8
    print('V4_CONFIG_CONTRACT_OK: soft pooling; semantic dropout=.10; margin=.15; adaptive weighted mean')
    return cfg


def smoke(cfg):
    import numpy as np
    import tensorflow as tf
    from train import build_model, compute_loss, configure_tensorflow_runtime, get_class_names, split_variables
    from datasets.fer2013 import SplitRecords, make_dataset, _resolve_split_csv_dir
    from rafdb_pose_common import read_test, CLASSES
    from models.soft_regional_pooling import SoftRegionalPooling
    configure_tensorflow_runtime(cfg)
    assert tf.config.list_physical_devices('GPU'), 'HPC GPU required for real smoke'
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(cfg['seed']['random_seed'])
    # Check zero-init parity and non-detached input/score gradients in mixed precision.
    pool = SoftRegionalPooling()
    x = tf.Variable(tf.random.normal([2, 8, 14, 512]))
    with tf.GradientTape() as tape:
        pooled = pool(x)
        objective = tf.reduce_sum(tf.square(tf.cast(pooled, tf.float32)))
    grads = tape.gradient(objective, [x, *pool.trainable_variables])
    expected = tf.reduce_mean(tf.cast(x, pooled.dtype), [1, 2])
    np.testing.assert_allclose(pooled.numpy(), expected.numpy(), atol=2e-3, rtol=2e-2)
    for grad in grads:
        assert grad is not None and np.isfinite(grad.numpy()).all() and np.any(grad.numpy() != 0)
    # Read-only discovery: no source CSV remapping/writes by the normal collector.
    root = _resolve_split_csv_dir(Path(cfg['data']['data_path']))
    rows = {s: read_test(root, root/f'{s}.csv', expected=3068 if s == 'test' else 0)
            for s in ('train', 'val', 'test')}
    assert len(rows['train']) + len(rows['val']) == 12271
    sets = {s: {r['image_path'] for r in rs} for s, rs in rows.items()}
    assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    assert get_class_names(cfg) == CLASSES
    for split, rs in rows.items():
        assert {r['label'] for r in rs} == set(range(7))
        print(f'{split}: N={len(rs)} class_counts={np.bincount([r["label"] for r in rs], minlength=7).tolist()}')
    rs = rows['train']
    records = SplitRecords(np.array([r['image_path'] for r in rs], dtype=object),
                           np.array([r['label'] for r in rs]), np.arange(len(rs)), None)
    ds = make_dataset(records, cfg, split='train', training=True, replicas=1)
    features, labels = next(iter(ds))
    assert tuple(features['image'].shape) == (cfg['runtime']['batch_size_per_gpu'], 112, 112, 3)
    print(f'INPUT_CONTRACT_OK shape={features["image"].shape} classes={CLASSES} labels={labels.numpy()}')
    # Two images suffice for the expensive gradient check. The dataset batch above
    # still verifies the configured training batch size without taking an update.
    features, labels = {k:v[:2] for k,v in features.items()}, labels[:2]
    model = build_model(cfg)
    model(features, training=False)
    assert model.use_soft_regional_pooling and model.semantic_projector_dropout == .1
    for name in ('visual_projector', 'visual_projector_upper', 'visual_projector_lower', 'visual_projector_au'):
        assert getattr(model, name).get_layer('drop').rate == .1
    backbone_vars, head_vars = split_variables(model)
    pool_vars = [v for layer in (model.soft_pool_upper, model.soft_pool_lower, model.soft_pool_au)
                 for v in layer.trainable_variables]
    assert len(pool_vars) == 3 and all(any(v is h for h in head_vars) for v in pool_vars)
    # Read-only forward/backward: no optimizer step, no checkpoint restore/write.
    with tf.GradientTape() as tape:
        output = model(features, training=False)
        total, parts = compute_loss(output, labels, cfg, model=model)
        semantic = parts['semantic']
    grads = tape.gradient(semantic, [backbone_vars[0], *pool_vars,
                                   model.visual_projector.trainable_variables[0],
                                   model.granularity_gate.trainable_variables[0]])
    for grad in grads:
        assert grad is not None and np.isfinite(grad.numpy()).all() and np.any(grad.numpy() != 0)
    assert np.isfinite(float(total))
    assert output['logits'].shape == output['semantic_logits'].shape == (2, 7)
    np.testing.assert_allclose(output['semantic_logits'].numpy(), output['agg_sim'].numpy()*20, atol=1e-5)
    np.testing.assert_allclose(output['granularity_weights'].numpy().sum(1), 1, atol=1e-3)
    print('V4_REAL_SMOKE_OK: real data, forward/loss, soft-pool/head/backbone semantic gradients; no training update')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'config_rafdb_siglip2_semantic_stable_v4_combined.yaml'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = contract(args.config)
    if args.smoke:
        smoke(cfg)
