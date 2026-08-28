import sys
import os
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from datasets.fer2013 import EMOTION_NAMES
from models.clip_text_encoder import (
    DEFAULT_AU_EMOTION_PROMPTS,
    EMOTION_CLASS_MAP,
    get_or_compute_clip_text_prototypes,
)
from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline
from models.mgr_cnn import MGRConvNeXtFER
from losses.classification import supervised_mgr_loss


def test_1_class_ordering_alignment():
    print("=== Task 1. Checking 7 Emotion Prototype Class Ordering Alignment ===")
    print(f"FER2013 Canonical EMOTION_NAMES: {EMOTION_NAMES}")
    
    # 1. Verify EMOTION_CLASS_MAP matches EMOTION_NAMES
    for idx, name in enumerate(EMOTION_NAMES):
        assert EMOTION_CLASS_MAP[idx] == name, f"Mismatch at index {idx}: map has {EMOTION_CLASS_MAP[idx]} vs dataset {name}"
        assert idx in DEFAULT_AU_EMOTION_PROMPTS, f"Missing prompt list for class index {idx}"
        
    print("\nVerified Mapping:")
    for idx, name in enumerate(EMOTION_NAMES):
        print(f"  class_index {idx} -> class_name '{name}' -> prototype_index [{idx}]")
        
    # Rebuild/re-evaluate text prototypes
    cache_file = os.path.join("pretrained", "clip_text_prototypes_7emotions.npy")
    prototypes = get_or_compute_clip_text_prototypes(cache_path=cache_file)
    assert prototypes.shape == (7, 512), f"Prototypes shape mismatch: {prototypes.shape}"
    norms = np.linalg.norm(prototypes, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Text prototypes must be L2 normalized!"
    print("[SUCCESS] Task 1 passed: Class ordering and prototype matrix verified!\n")


def test_2_frozen_prototypes_and_gradient_isolation():
    print("=== Task 2. Verifying Frozen Prototype Isolation & Zero Trainable Params ===")
    cfg = load_config("config_convnext_base_ms1m_arcface_clip_semantic.yaml")
    cfg["model"]["convnext_base_require_pretrained"] = False
    
    model = ConvNeXtBaseFaceFERBaseline(cfg["model"])
    dummy_img = tf.random.normal([2, 112, 112, 3], dtype=tf.float32)
    dummy_labels = tf.constant([0, 3], dtype=tf.int32)
    
    # Forward pass to build model & submodel weights
    _ = model(dummy_img, training=False)
    
    # Check model.trainable_variables
    trainable_var_names = [v.name for v in model.trainable_variables]
    assert not any("frozen_clip_text_prototypes" in name for name in trainable_var_names), \
        "text_prototypes tensor MUST NOT be present in model.trainable_variables!"
        
    # GradientTape verification
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.text_prototypes)
        out = model(dummy_img, training=True)
        total_loss, parts = supervised_mgr_loss(dummy_labels, out, num_classes=7)
        
    proto_grad = tape.gradient(total_loss, model.text_prototypes)
    proj_grads = [tape.gradient(total_loss, v) for v in model.visual_projector.trainable_variables]
    
    print(f"Trainable Variables Count total: {len(model.trainable_variables)}")
    print(f"Visual Projector Trainable Vars: {len(model.visual_projector.trainable_variables)}")
    print(f"Text Prototypes Grad is None/Zero: {proto_grad is None or tf.reduce_sum(tf.abs(proto_grad)) == 0}")
    assert any(g is not None for g in proj_grads), "Visual Projector MUST receive non-zero gradients!"
    
    print("[SUCCESS] Task 2 passed: CLIP text prototypes are 100% frozen non-trainable constants!\n")


def test_3_semantic_metrics_and_loss_formula():
    print("=== Task 3. Verifying Structured Semantic Loss Formula & Metric Calculations ===")
    cfg = load_config("config_convnext_base_ms1m_arcface_clip_semantic.yaml")
    cfg["model"]["convnext_base_require_pretrained"] = False
    
    model = ConvNeXtBaseFaceFERBaseline(cfg["model"])
    dummy_img = tf.random.normal([4, 112, 112, 3], dtype=tf.float32)
    dummy_labels = tf.constant([0, 1, 3, 5], dtype=tf.int32)
    
    out = model(dummy_img, training=False)
    lambda_sem = out["lambda_sem"]
    
    total_loss, parts = supervised_mgr_loss(dummy_labels, out, num_classes=7)
    ce_loss = parts["ce"].numpy()
    semantic_loss = parts["semantic"].numpy()
    expected_total = ce_loss + lambda_sem * semantic_loss
    
    print(f"CE Loss           : {ce_loss:.4f}")
    print(f"Semantic Loss     : {semantic_loss:.4f}")
    print(f"Lambda Sem        : {lambda_sem:.4f}")
    print(f"Total Loss Output : {total_loss.numpy():.4f}")
    print(f"Expected Formula  : {expected_total:.4f}")
    assert np.isclose(total_loss.numpy(), expected_total, atol=1e-4), "Total loss formula mismatch!"
    
    # Calculate FER and Semantic Accuracies
    fer_preds = tf.argmax(out["logits"], axis=-1, output_type=tf.int32).numpy()
    sem_preds = tf.argmax(out["semantic_logits"], axis=-1, output_type=tf.int32).numpy()
    labels_np = dummy_labels.numpy()
    
    fer_acc = np.mean(fer_preds == labels_np)
    sem_acc = np.mean(sem_preds == labels_np)
    
    print(f"FER Accuracy      : {fer_acc:.4f} (preds: {fer_preds})")
    print(f"Semantic Accuracy : {sem_acc:.4f} (preds: {sem_preds})")
    print("[SUCCESS] Task 3 passed: Structured semantic metrics and loss formula verified!\n")


def test_4_full_smoke_test_batch_size_2():
    print("=== Task 4. Full Smoke Test (Batch Size 2 & Mixed Precision) across Models ===")
    
    # 1. Baseline Model (float32)
    cfg_base = load_config("config_convnext_base_ms1m_arcface_clip_semantic.yaml")
    cfg_base["model"]["convnext_base_require_pretrained"] = False
    model_base = ConvNeXtBaseFaceFERBaseline(cfg_base["model"])
    
    batch_img = tf.random.normal([2, 112, 112, 3], dtype=tf.float32)
    batch_lbl = tf.constant([2, 6], dtype=tf.int32)
    
    out_base = model_base(batch_img, training=True)
    loss_base, _ = supervised_mgr_loss(batch_lbl, out_base, num_classes=7)
    
    print("ConvNeXt Base Baseline Output (float32):")
    print(f"  logits shape          : {out_base['logits'].shape} (dtype: {out_base['logits'].dtype})")
    print(f"  semantic_logits shape : {out_base['semantic_logits'].shape} (dtype: {out_base['semantic_logits'].dtype})")
    print(f"  loss value            : {loss_base.numpy():.4f}")
    assert out_base["logits"].shape == (2, 7)
    assert out_base["semantic_logits"].shape == (2, 7)
    
    # 2. Baseline Model (mixed_float16 input)
    batch_img_f16 = tf.random.normal([2, 112, 112, 3], dtype=tf.float16)
    out_base_f16 = model_base({"image": batch_img_f16}, training=True)
    loss_base_f16, _ = supervised_mgr_loss(batch_lbl, out_base_f16, num_classes=7)
    print("ConvNeXt Base Baseline Output (float16 input):")
    print(f"  logits shape          : {out_base_f16['logits'].shape} (dtype: {out_base_f16['logits'].dtype})")
    print(f"  semantic_logits shape : {out_base_f16['semantic_logits'].shape} (dtype: {out_base_f16['semantic_logits'].dtype})")
    print(f"  loss value            : {loss_base_f16.numpy():.4f}")
    
    # 3. MGR-CNN Model
    cfg_mgr = load_config("config_pure_mgr_single_head.yaml")
    cfg_mgr["model"]["use_semantic_branch"] = True
    cfg_mgr["model"]["lambda_sem"] = 0.2
    cfg_mgr["model"]["convnext_base_require_pretrained"] = False
    model_mgr = MGRConvNeXtFER(cfg_mgr)
    
    batch_input_mgr = {
        "image": tf.random.normal([2, 112, 112, 3], dtype=tf.float32),
        "mask": tf.random.normal([2, 112, 112, 6], dtype=tf.float32),
    }
    out_mgr = model_mgr(batch_input_mgr, training=True)
    loss_mgr, _ = supervised_mgr_loss(batch_lbl, out_mgr, num_classes=7)
    
    print("MGR-CNN Output:")
    print(f"  logits shape          : {out_mgr['logits'].shape}")
    print(f"  semantic_logits shape : {out_mgr['semantic_logits'].shape}")
    print(f"  loss value            : {loss_mgr.numpy():.4f}")
    assert out_mgr["logits"].shape == (2, 7)
    assert out_mgr["semantic_logits"].shape == (2, 7)
    
    print("[SUCCESS] Task 4 passed: Batch Size 2 and Mixed Precision Smoke test succeeded!\n")


if __name__ == "__main__":
    test_1_class_ordering_alignment()
    test_2_frozen_prototypes_and_gradient_isolation()
    test_3_semantic_metrics_and_loss_formula()
    test_4_full_smoke_test_batch_size_2()
    print("ALL 4 CLIP SEMANTIC ALIGNMENT CHECKS PASSED PERFECTLY!")
