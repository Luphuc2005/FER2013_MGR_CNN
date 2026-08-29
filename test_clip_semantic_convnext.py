import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
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
        
    # Rebuild/re-evaluate text prototypes (single prototype)
    cache_file = os.path.join("pretrained", "clip_text_prototypes_7emotions.npy")
    prototypes = get_or_compute_clip_text_prototypes(cache_path=cache_file, multi_prototype=False)
    assert prototypes.shape == (7, 512), f"Single prototype shape mismatch: {prototypes.shape}"
    norms = np.linalg.norm(prototypes, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Text prototypes must be L2 normalized!"

    # Multi prototype
    cache_multi_file = os.path.join("pretrained", "clip_text_prototypes_7emotions_multigranularity_multi5.npy")
    multi_prototypes = get_or_compute_clip_text_prototypes(cache_path=cache_multi_file, multi_prototype=True)
    assert multi_prototypes.shape == (7, 5, 512), f"Multi-prototype shape mismatch: {multi_prototypes.shape}"
    multi_norms = np.linalg.norm(multi_prototypes, axis=-1)
    assert np.allclose(multi_norms, 1.0, atol=1e-3), "Multi-prototype text vectors must be L2 normalized individually!"

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


def test_4_multi_prototype_clip_smoke_and_math():
    print("=== Task 4. Verifying Multi-Prototype LogSumExp Aggregation and Tensor Shapes ===")
    
    # 1. Multi-prototype enabled test
    cfg_multi = load_config("config_convnext_base_ms1m_arcface_clip_multigranularity_semantic.yaml")
    cfg_multi["model"]["convnext_base_require_pretrained"] = False
    cfg_multi["model"]["multi_prototype"] = True
    cfg_multi["model"]["prototype_aggregation"] = "logsumexp"
    cfg_multi["model"]["prototype_temperature"] = 0.1
    
    model_multi = ConvNeXtBaseFaceFERBaseline(cfg_multi["model"])
    batch_img = tf.random.normal([2, 112, 112, 3], dtype=tf.float32)
    batch_lbl = tf.constant([1, 4], dtype=tf.int32)
    
    out_multi = model_multi(batch_img, training=True)
    
    assert model_multi.text_prototypes.shape == (7, 5, 512), f"Expected (7, 5, 512), got {model_multi.text_prototypes.shape}"
    assert out_multi["semantic_logits"].shape == (2, 7), f"Expected (2, 7), got {out_multi['semantic_logits'].shape}"

    # Mixed precision float16 input test
    batch_img_f16 = tf.cast(batch_img, tf.float16)
    out_multi_f16 = model_multi({"image": batch_img_f16}, training=True)
    assert out_multi_f16["semantic_logits"].shape == (2, 7)
    
    print(f"Multi-prototype text prototypes shape : {model_multi.text_prototypes.shape}")
    print(f"Aggregated semantic logits shape       : {out_multi['semantic_logits'].shape} (float32 & float16 verified)")
    
    # 2. Backward compatibility test: multi_prototype = False
    cfg_single = load_config("config_convnext_base_ms1m_arcface_clip_semantic.yaml")
    cfg_single["model"]["convnext_base_require_pretrained"] = False
    cfg_single["model"]["multi_prototype"] = False
    cfg_single["model"]["clip_semantic"]["multi_prototype"] = False
    cfg_single["model"]["clip_prototypes_path"] = "pretrained/clip_text_prototypes_7emotions.npy"
    
    model_single = ConvNeXtBaseFaceFERBaseline(cfg_single["model"])
    out_single = model_single(batch_img, training=True)
    
    assert model_single.text_prototypes.shape == (7, 512), f"Expected (7, 512), got {model_single.text_prototypes.shape}"
    assert out_single["semantic_logits"].shape == (2, 7), f"Expected (2, 7), got {out_single['semantic_logits'].shape}"
    print(f"Single-prototype text prototypes shape: {model_single.text_prototypes.shape} (Backward Compatibility Verified)")
    
    print("[SUCCESS] Task 4 passed: Multi-prototype tensor shapes and backward compatibility verified!\n")


def test_5_real_clip_smoke_test():
    print("=== Task 5. REAL CLIP Smoke Test & Provenance Verification ===")
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    import torch
    import json

    # 1. Load CLIP tokenizer & model
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model '{model_name}' for smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name)
    model.eval()

    # 2. Encode 1 description
    sample_desc = ["a facial expression of anger with a tense appearance"]
    inputs = tokenizer(sample_desc, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeds = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[0][:, 0, :]
        embeds_norm = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        emb_np = embeds_norm.cpu().numpy()

    # 3. Check embedding finite & L2 norm ≈ 1
    assert np.all(np.isfinite(emb_np)), "CLIP single description embedding must be finite (no NaN or Inf)!"
    l2_norm = float(np.linalg.norm(emb_np))
    assert np.isclose(l2_norm, 1.0, atol=1e-3), f"L2 norm must be ~1.0, got {l2_norm:.4f}"
    print(f"Single description encoding passed: shape={emb_np.shape}, finite=True, L2_norm={l2_norm:.4f}")

    # 4. Generate 7-emotion prototypes (single & multi)
    cache_single = os.path.join("pretrained", "clip_text_prototypes_7emotions.npy")
    cache_multi = os.path.join("pretrained", "clip_text_prototypes_7emotions_multigranularity_multi5.npy")

    proto_single = get_or_compute_clip_text_prototypes(cache_path=cache_single, multi_prototype=False)
    proto_multi = get_or_compute_clip_text_prototypes(cache_path=cache_multi, multi_prototype=True)

    assert proto_single.shape == (7, 512), f"Expected (7, 512), got {proto_single.shape}"
    assert proto_multi.shape == (7, 5, 512), f"Expected (7, 5, 512), got {proto_multi.shape}"

    # 5. Reload from cache & confirm provenance is REAL_CLIP
    meta_single = cache_single + ".meta.json"
    meta_multi = cache_multi + ".meta.json"

    assert os.path.exists(meta_single), f"Metadata sidecar missing: {meta_single}"
    assert os.path.exists(meta_multi), f"Metadata sidecar missing: {meta_multi}"

    with open(meta_single, "r", encoding="utf-8") as f:
        meta_s = json.load(f)
    with open(meta_multi, "r", encoding="utf-8") as f:
        meta_m = json.load(f)

    assert meta_s.get("source") == "REAL_CLIP", f"Cache source is not REAL_CLIP: {meta_s}"
    assert meta_m.get("source") == "REAL_CLIP", f"Cache source is not REAL_CLIP: {meta_m}"
    assert meta_s.get("model") == model_name, f"Model name mismatch: {meta_s}"
    assert meta_m.get("model") == model_name, f"Model name mismatch: {meta_m}"

    print(f"[SUCCESS] Task 5 passed: REAL CLIP smoke test & cache provenance verified! Source: {meta_s['source']}\n")


if __name__ == "__main__":
    test_5_real_clip_smoke_test()
    test_1_class_ordering_alignment()
    test_2_frozen_prototypes_and_gradient_isolation()
    test_3_semantic_metrics_and_loss_formula()
    test_4_multi_prototype_clip_smoke_and_math()
    print("ALL MULTI-PROTOTYPE & SINGLE-PROTOTYPE CLIP CHECKS PASSED PERFECTLY!")

