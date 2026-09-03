from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional
import numpy as np

# Multi-Granularity Prompt Bank for 7 FER emotions (P1: Emotion, P2: AU, P3: Upper-face, P4: Lower-face, P5: Combined)
# Contextual Ensemble & Hard-Pair Discriminative Prompt Bank (3 Templates per Granularity Level)
ENSEMBLE_DISCRIMINATIVE_PROMPTS: Dict[int, List[List[str]]] = {
    0: [  # 0: angry
        # P1: Emotion-level (Ensemble 3 templates)
        [
            "a facial expression of anger with a tense and hostile appearance",
            "a close-up portrait of a person showing intense anger and aggression",
            "a face with an aggressive angry expression and glaring look",
        ],
        # P2: AU-level Discriminative
        [
            "anger characterized by brow lowerer AU4, lid tightener AU7, and lip tightener AU23",
            "facial action units of anger with brow lowerer AU4 and tightly pressed lips AU23",
            "distinctive anger facial cues with lowered eyebrows AU4 and compressed lips AU23",
        ],
        # P3: Upper-face Discriminative (vs Sad slanted brows)
        [
            "strongly lowered and furrowed eyebrows drawn down with tense narrowed eyes and an intense stare",
            "furrowed brows and glaring narrowed eyes with forehead tension, distinct from sad slanted brows",
            "intense glare with lowered eyebrows pulled inward and downward",
        ],
        # P4: Lower-face Discriminative (vs Sad mouth corners)
        [
            "firmly pressed or tightened lips with visible tension around mouth and jaw, without drooping",
            "tightly closed pressed lips with clenched jaw and no mouth corner depression",
            "tense mouth with lips firmly pressed together and firm chin",
        ],
        # P5: Combined
        [
            "an angry face with lowered brows, narrowed intense eyes, and tightly pressed lips, consistent with AU4, AU7, and AU23",
            "a complete facial portrait of anger with furrowed brows, tense eyes, and pressed lips",
            "hostile angry face featuring lowered eyebrows, glaring eyes, and firm tight lips",
        ],
    ],
    1: [  # 1: disgust
        # P1: Emotion-level
        [
            "a facial expression of disgust showing strong aversion and distaste",
            "a close-up facial portrait of a person displaying repulse and revulsion",
            "a face showing severe distaste and repulsive expression",
        ],
        # P2: AU-level
        [
            "disgust characterized by nose wrinkler AU9, upper lip raiser AU10, and lip corner depressor AU15",
            "action units of disgust with nose wrinkler AU9 and sneering upper lip AU10",
            "facial cues of disgust featuring wrinkled nose AU9 and raised upper lip AU10",
        ],
        # P3: Upper-face
        [
            "wrinkled nose bridge with squinted eyes, lowered brows, and crinkled lower eyelids",
            "strongly crinkled nose bridge between squinted narrowed eyes",
            "squinted eyes with wrinkled nose bridge and slightly lowered brows",
        ],
        # P4: Lower-face
        [
            "raised upper lip sneering upward with turned-down mouth corners and a raised chin",
            "sneering raised upper lip exposing upper teeth with downturned mouth corners",
            "asymmetric raised upper lip sneer with turned-down mouth corners",
        ],
        # P5: Combined
        [
            "a disgusted face with a wrinkled nose, sneering raised upper lip, and downturned mouth corners, consistent with AU9, AU10, and AU15",
            "a full facial expression of disgust featuring a wrinkled nose bridge and sneering lip",
            "distasted face showing nose wrinkling and upper lip sneering upward",
        ],
    ],
    2: [  # 2: fear
        # P1: Emotion-level
        [
            "a facial expression of fear with a terrified and alarmed look",
            "a close-up facial crop of a person displaying intense fear and panic",
            "a terrified face showing fright, panic, and alarm",
        ],
        # P2: AU-level Discriminative (vs Surprise AU26)
        [
            "fear characterized by inner brow raiser AU1, outer brow raiser AU2, upper lid raiser AU5, and lip stretcher AU20",
            "fear action units with raised inner brows AU1+AU2, wide lid raiser AU5, and lip stretcher AU20",
            "fear facial cues featuring high inner brows AU1+AU2 and horizontally stretched lips AU20",
        ],
        # P3: Upper-face Discriminative
        [
            "raised and drawn together inner eyebrows with wide open startled eyes displaying white sclera",
            "high raised inner brows drawn together in panic with wide staring un-squinted eyes",
            "startled wide eyes with raised inner eyebrows, distinct from relaxed surprise brows",
        ],
        # P4: Lower-face Discriminative (vs Surprise oval jaw)
        [
            "horizontally stretched lips pulled back toward ears with a tense, slightly open mouth",
            "lips pulled sideways horizontally toward ears with mouth tension, distinct from oval dropped jaw",
            "horizontally stretched open mouth with lips pulled back flat toward cheeks",
        ],
        # P5: Combined
        [
            "a fearful face with high raised brows, wide open frightened eyes, and horizontally stretched lips, consistent with AU1, AU2, AU5, and AU20",
            "a panicked face featuring pulled-up inner brows, wide frightened eyes, and horizontally stretched mouth",
            "terrified expression with wide open sclera-visible eyes and sideways stretched lips",
        ],
    ],
    3: [  # 3: happy
        # P1: Emotion-level
        [
            "a facial expression of happiness with a joyful and cheerful appearance",
            "a close-up portrait of a person beaming with genuine happiness and joy",
            "a cheerful face displaying a warm happy smile",
        ],
        # P2: AU-level
        [
            "happiness characterized by cheek raiser AU6 and lip corner puller AU12",
            "duchenne smile action units featuring cheek raiser AU6 and lip corner puller AU12",
            "happy facial cues with raised cheeks AU6 and upward smiling mouth AU12",
        ],
        # P3: Upper-face
        [
            "raised cheeks with crinkled eye corners forming visible crow's feet and warm smiling eyes",
            "crinkled eyes with crow's feet wrinkles and raised cheeks in a warm smile",
            "smiling eyes with raised cheeks and visible eye corner wrinkles",
        ],
        # P4: Lower-face
        [
            "lip corners pulled upward and backward in a broad smile showing visible teeth",
            "broad upward smiling lips with pulled mouth corners exposing teeth",
            "wide open smile with corners of the mouth turned strongly upward",
        ],
        # P5: Combined
        [
            "a happy face with broad upward smiling lips, raised cheeks, and crinkled smiling eyes, consistent with AU6 and AU12",
            "a joyful beaming expression with upward smiling lips and crinkled eyes",
            "genuine duchenne smile featuring raised cheeks, eye wrinkles, and upward mouth corners",
        ],
    ],
    4: [  # 4: sad
        # P1: Emotion-level
        [
            "a facial expression of sadness with a somber and sorrowful look",
            "a close-up facial crop of a person displaying grief, sorrow, and sadness",
            "a mournful face showing sorrow and deep sadness",
        ],
        # P2: AU-level Discriminative (vs Anger AU4/AU23)
        [
            "sadness characterized by inner brow raiser AU1, brow lowerer AU4, and lip corner depressor AU15",
            "action units of sadness featuring inner brow raiser AU1, AU4, and downturned lip corners AU15",
            "sad facial cues with slanted inner eyebrows AU1+AU4 and depressed mouth corners AU15",
        ],
        # P3: Upper-face Discriminative
        [
            "inner corners of the eyebrows raised and slanted upward with drooping upper eyelids and dull eyes",
            "slanted eyebrows with inner corners raised upward in sorrow and drooping eyelids, distinct from anger tension",
            "sorrowful eyes with slanted raised inner brows and drooping eyelids",
        ],
        # P4: Lower-face Discriminative
        [
            "downturned lip corners with a trembling or pouty mouth and loose jaw",
            "turned-down mouth corners with pouty lower lip and depressed lip corners AU15",
            "drooping mouth corners with a loose chin and sorrowful lips",
        ],
        # P5: Combined
        [
            "a sad face with slanted raised inner brows, drooping eyelids, and downturned mouth corners, consistent with AU1, AU4, and AU15",
            "a sorrowful face featuring slanted inner brows, dull drooping eyes, and downturned lips",
            "somber expression showing raised inner eyebrows, drooping eyelids, and depressed mouth corners",
        ],
    ],
    5: [  # 5: surprise
        # P1: Emotion-level
        [
            "a facial expression of surprise with an astonished and amazed appearance",
            "a close-up portrait of a person displaying shock, amazement, and surprise",
            "an astonished face with an amazed and shocked expression",
        ],
        # P2: AU-level Discriminative (vs Fear AU20)
        [
            "surprise characterized by inner brow raiser AU1, outer brow raiser AU2, upper lid raiser AU5, and jaw drop AU26",
            "action units of surprise with high brow raiser AU1+AU2 and dropped jaw AU26",
            "surprise facial cues featuring high arched brows AU1+AU2 and relaxed open jaw AU26",
        ],
        # P3: Upper-face Discriminative
        [
            "high curved eyebrows raised up the forehead with widely opened round un-squinted eyes",
            "high arched smooth eyebrows raised up without center furrowing and open round eyes",
            "round open staring eyes with high arched raised eyebrows",
        ],
        # P4: Lower-face Discriminative (vs Fear stretched lips)
        [
            "dropped jaw creating an oval-shaped open mouth with relaxed un-stretched lips",
            "relaxed open jaw forming an oval mouth posture, distinct from horizontally stretched fear lips",
            "oval dropped open jaw with relaxed mouth and no horizontal tension",
        ],
        # P5: Combined
        [
            "a surprised face with high arched brows, widely opened staring eyes, and a dropped open jaw, consistent with AU1, AU2, AU5, and AU26",
            "an astonished face featuring high arched brows, round open eyes, and an oval dropped jaw",
            "shocked expression showing high raised eyebrows, open staring eyes, and relaxed open mouth",
        ],
    ],
    6: [  # 6: neutral
        # P1: Emotion-level
        [
            "a calm and neutral facial expression without emotion",
            "a close-up portrait of a person with a serene and expressionless face",
            "an emotionless neutral human face with a calm look",
        ],
        # P2: AU-level
        [
            "a neutral face characterized by the absence of active action units",
            "absence of facial action unit activation in a calm neutral state",
            "resting facial posture with zero active facial muscle contraction",
        ],
        # P3: Upper-face
        [
            "relaxed eyebrows and smooth forehead with calm resting eyes",
            "smooth resting forehead with relaxed un-furrowed brows and calm eyes",
            "un-wrinkled smooth forehead with resting calm eyes",
        ],
        # P4: Lower-face
        [
            "closed relaxed lips in a natural resting posture with no tension in the mouth or jaw",
            "natural closed lips in resting position with no tension around mouth",
            "relaxed closed mouth with neutral jaw position",
        ],
        # P5: Combined
        [
            "a neutral facial expression with balanced resting features, smooth brows, calm eyes, and closed relaxed lips",
            "a serene face displaying calm un-furrowed brows, resting eyes, and closed lips",
            "balanced neutral expression with smooth features and relaxed mouth",
        ],
    ],
}


EMOTION_CLASS_MAP: Dict[int, str] = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def compute_prompt_hash(prompts_per_class: Dict[int, Any]) -> tuple[str, int]:
    """
    Computes a deterministic SHA256 hash and total count for a prompt dictionary.
    Keys are sorted numerically and prompt lists are serialized predictably.
    """
    sorted_data = {str(k): prompts_per_class[k] for k in sorted(prompts_per_class.keys())}
    json_str = json.dumps(sorted_data, sort_keys=True, ensure_ascii=True)
    prompt_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    prompt_count = 0
    for v in prompts_per_class.values():
        for item in v:
            if isinstance(item, list):
                prompt_count += len(item)
            else:
                prompt_count += 1
    return prompt_hash, prompt_count


def validate_cache_provenance(
    cache_path: str,
    meta_path: str,
    model_name: str,
    expected_shape: tuple,
    expected_prompt_hash: Optional[str] = None,
    expected_prompt_count: Optional[int] = None,
) -> bool:
    """Checks if cache file and metadata sidecar exist and are valid REAL_CLIP prototypes."""
    if not os.path.exists(cache_path) or not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("source") != "REAL_CLIP":
            return False
        if meta.get("model") != model_name:
            return False
        if tuple(meta.get("shape", [])) != expected_shape:
            return False

        # Validate prompt hash & count for content integrity
        if expected_prompt_hash is not None:
            if meta.get("prompt_hash") != expected_prompt_hash:
                return False
        if expected_prompt_count is not None:
            if meta.get("prompt_count") != expected_prompt_count:
                return False

        prototypes = np.load(cache_path)
        if prototypes.shape != expected_shape:
            return False
        if not np.all(np.isfinite(prototypes)):
            return False

        norms = np.linalg.norm(prototypes, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            return False

        return True
    except Exception:
        return False


def get_or_compute_clip_text_prototypes(
    model_name: str = "openai/clip-vit-base-patch32",
    cache_path: Optional[str] = None,
    prompts_per_class: Optional[Dict[int, Any]] = None,
    embedding_dim: int = 512,
    multi_prototype: bool = False,
) -> np.ndarray:
    """
    Computes or loads text prototypes for 7 FER emotion classes using a frozen REAL CLIP text encoder.
    Synthetic/random prototype fallbacks are COMPLETELY DISABLED.
    If loading CLIP fails, raises RuntimeError immediately.
    """
    if prompts_per_class is None:
        prompts_per_class = ENSEMBLE_DISCRIMINATIVE_PROMPTS

    prompt_hash, prompt_count = compute_prompt_hash(prompts_per_class)

    if cache_path is None or ("clip_text_prototypes_7emotions.npy" in cache_path and "siglip" in model_name.lower()):
        safe_model_tag = "siglip2" if "siglip2" in model_name.lower() else ("siglip" if "siglip" in model_name.lower() else "clip")
        cache_path = (
            f"pretrained/{safe_model_tag}_text_prototypes_7emotions_multigranularity_multi5.npy"
            if multi_prototype
            else f"pretrained/{safe_model_tag}_text_prototypes_7emotions.npy"
        )

    meta_path = cache_path + ".meta.json"
    expected_shape = (7, 5, embedding_dim) if multi_prototype else (7, embedding_dim)

    # Validate existing cache & provenance (including prompt content hash)
    if validate_cache_provenance(
        cache_path,
        meta_path,
        model_name,
        expected_shape,
        expected_prompt_hash=prompt_hash,
        expected_prompt_count=prompt_count,
    ):
        prototypes = np.load(cache_path).astype(np.float32)
        print(f"[CLIP/SigLIP] Model: {model_name}", flush=True)
        print(f"[CLIP/SigLIP] Text encoder loaded successfully", flush=True)
        print(f"[CLIP/SigLIP] Prototype source: REAL_CLIP/SIGLIP", flush=True)
        print(f"[CLIP/SigLIP] Prompt SHA256 Hash: {prompt_hash[:16]}... (count: {prompt_count})", flush=True)
        print(f"[CLIP/SigLIP] Synthetic fallback: DISABLED", flush=True)
        print(f"[CLIP/SigLIP] Prototype shape: {prototypes.shape}", flush=True)
        print(f"[CLIP/SigLIP] L2 norm check: PASSED (loaded from verified cache: {cache_path})", flush=True)
        return prototypes

    if os.path.exists(cache_path):
        print(f"[CLIP/SigLIP] Stale, modified prompt, or unverified cache detected at '{cache_path}'. Regenerating using REAL model...", flush=True)
        try:
            os.remove(cache_path)
        except Exception:
            pass
    if os.path.exists(meta_path):
        try:
            os.remove(meta_path)
        except Exception:
            pass

    # Encode using REAL Text Encoder via HuggingFace Transformers
    prototypes = _encode_with_transformers(
        model_name=model_name,
        prompts_per_class=prompts_per_class,
        embedding_dim=embedding_dim,
        multi_prototype=multi_prototype,
    )

    if prototypes is None:
        raise RuntimeError(
            f"[CLIP/SigLIP ERROR] Could not initialize or run HuggingFace model '{model_name}'.\n"
            f"Synthetic fallback is DISABLED. Training cannot proceed without real text embeddings!\n"
            f"Possible causes:\n"
            f"  1) Compute node has no internet access to download HuggingFace model.\n"
            f"     --> FIX: Upload/copy the pre-generated 'pretrained/*.npy' and 'pretrained/*.meta.json' cache files from local to server.\n"
            f"  2) Thư viện 'transformers' hoặc 'torch' chưa được cài trong environment."
        )

    # Post-checks
    if prototypes.shape != expected_shape:
        raise RuntimeError(
            f"[CLIP/SigLIP ERROR] Expected prototype shape {expected_shape}, but got {prototypes.shape}."
        )

    if not np.all(np.isfinite(prototypes)):
        raise RuntimeError("[CLIP/SigLIP ERROR] Generated text prototypes contain non-finite values (NaN or Inf).")

    norms = np.linalg.norm(prototypes, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(
            f"[CLIP/SigLIP ERROR] Generated text prototypes failed L2 normalization check (min norm={norms.min():.4f}, max norm={norms.max():.4f})."
        )

    # Save cache & provenance metadata including prompt_hash & prompt_count
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            np.save(cache_path, prototypes)
            meta_data = {
                "model": model_name,
                "source": "REAL_CLIP",
                "embedding_dim": embedding_dim,
                "multi_prototype": multi_prototype,
                "shape": list(prototypes.shape),
                "prompt_hash": prompt_hash,
                "prompt_count": prompt_count,
                "l2_normalized": True,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)
            print(f"[CLIP/SigLIP] Cached prototypes saved to {cache_path}", flush=True)
        except Exception as e:
            print(f"[CLIP/SigLIP] Warning: Could not save text prototypes cache: {e}", flush=True)

    print(f"[CLIP/SigLIP] Model: {model_name}", flush=True)
    print(f"[CLIP/SigLIP] Text encoder loaded successfully", flush=True)
    print(f"[CLIP/SigLIP] Prototype source: REAL_CLIP/SIGLIP", flush=True)
    print(f"[CLIP/SigLIP] Prompt SHA256 Hash: {prompt_hash[:16]}... (count: {prompt_count})", flush=True)
    print(f"[CLIP/SigLIP] Synthetic fallback: DISABLED", flush=True)
    print(f"[CLIP/SigLIP] Prototype shape: {prototypes.shape}", flush=True)
    print(f"[CLIP/SigLIP] L2 norm check: PASSED (all vectors L2 norm ≈ 1.0)", flush=True)

    return prototypes.astype(np.float32)


def _encode_with_transformers(
    model_name: str,
    prompts_per_class: Dict[int, List[str]],
    embedding_dim: int,
    multi_prototype: bool = False,
) -> Optional[np.ndarray]:
    """Encodes text prompts using HuggingFace transformers CLIP/SigLIP text model (PyTorch or TF)."""
    # 1. Try PyTorch Transformers
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel, CLIPTextModelWithProjection, CLIPModel

        def _extract_tensor(out):
            if isinstance(out, torch.Tensor):
                return out
            if hasattr(out, "text_embeds") and isinstance(out.text_embeds, torch.Tensor):
                return out.text_embeds
            if hasattr(out, "pooler_output") and isinstance(out.pooler_output, torch.Tensor):
                return out.pooler_output
            if hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
                return out.last_hidden_state[:, 0, :]
            if isinstance(out, (list, tuple)) and len(out) > 0:
                return _extract_tensor(out[0])
            raise TypeError(f"Cannot extract torch.Tensor from HuggingFace output of type {type(out)}")

        print(f"[CLIP/SigLIP] Loading PyTorch HuggingFace model '{model_name}' (multi_prototype={multi_prototype})...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder = None
        try:
            text_encoder = AutoModel.from_pretrained(model_name)
        except Exception:
            try:
                text_encoder = CLIPTextModelWithProjection.from_pretrained(model_name)
            except Exception:
                text_encoder = CLIPModel.from_pretrained(model_name)

        text_encoder.eval()

        class_prototypes = []
        with torch.no_grad():
            for c in range(7):
                p_items = prompts_per_class[c]
                if multi_prototype and len(p_items) == 5:
                    gran_embeds = []
                    for p_item in p_items:
                        prompts = [p_item] if isinstance(p_item, str) else p_item
                        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
                        if hasattr(text_encoder, "get_text_features"):
                            out = text_encoder.get_text_features(**inputs)
                        else:
                            out = text_encoder(**inputs)

                        embeds = _extract_tensor(out)
                        embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                        level_embed = embeds.mean(dim=0)
                        level_embed = level_embed / level_embed.norm(p=2, dim=-1, keepdim=True)
                        gran_embeds.append(level_embed.cpu().numpy())
                    class_prototypes.append(np.stack(gran_embeds, axis=0))
                else:
                    flat_prompts = []
                    for item in p_items:
                        if isinstance(item, list):
                            flat_prompts.extend(item)
                        else:
                            flat_prompts.append(item)
                    inputs = tokenizer(flat_prompts, padding=True, return_tensors="pt")
                    if hasattr(text_encoder, "get_text_features"):
                        out = text_encoder.get_text_features(**inputs)
                    else:
                        out = text_encoder(**inputs)

                    embeds = _extract_tensor(out)
                    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                    if multi_prototype:
                        class_prototypes.append(embeds.cpu().numpy())
                    else:
                        mean_embed = embeds.mean(dim=0)
                        mean_embed = mean_embed / mean_embed.norm(p=2, dim=-1, keepdim=True)
                        class_prototypes.append(mean_embed.cpu().numpy())

        res = np.stack(class_prototypes, axis=0)
        return res
    except Exception as e_pt:
        print(f"[CLIP] PyTorch encoding attempt: {e_pt}", flush=True)

    # 2. Try TensorFlow Transformers
    try:
        import tensorflow as tf
        from transformers import AutoTokenizer, TFCLIPModel, TFCLIPTextModel

        print(f"[CLIP] Loading TensorFlow HuggingFace model '{model_name}' (multi_prototype={multi_prototype})...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            text_encoder = TFCLIPModel.from_pretrained(model_name)
            is_full_clip = True
        except Exception:
            text_encoder = TFCLIPTextModel.from_pretrained(model_name)
            is_full_clip = False

        class_prototypes = []
        for c in range(7):
            prompts = prompts_per_class[c]
            inputs = tokenizer(prompts, padding=True, return_tensors="tf")
            if is_full_clip and hasattr(text_encoder, "get_text_features"):
                embeds = text_encoder.get_text_features(**inputs)
            else:
                outputs = text_encoder(**inputs)
                embeds = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, "last_hidden_state") else outputs[0][:, 0, :]

            embeds = embeds / tf.norm(embeds, ord=2, axis=-1, keepdims=True)
            if multi_prototype:
                class_prototypes.append(embeds.numpy())
            else:
                mean_embed = tf.reduce_mean(embeds, axis=0)
                mean_embed = mean_embed / tf.norm(mean_embed, ord=2, axis=-1, keepdims=True)
                class_prototypes.append(mean_embed.numpy())

        res = np.stack(class_prototypes, axis=0)
        return res
    except Exception as e_tf:
        print(f"[CLIP] TensorFlow encoding attempt: {e_tf}", flush=True)

    return None

