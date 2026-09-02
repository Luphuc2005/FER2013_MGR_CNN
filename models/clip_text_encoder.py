from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional
import numpy as np

# Multi-Granularity Prompt Bank for 7 FER emotions (P1: Emotion, P2: AU, P3: Upper-face, P4: Lower-face, P5: Combined)
DEFAULT_AU_EMOTION_PROMPTS: Dict[int, List[str]] = {
    0: [  # 0: angry
        # P1: Emotion-level
        "a facial expression of anger with a tense and hostile appearance",
        # P2: AU-level
        "anger characterized by brow lowerer AU4, lid tightener AU7, and lip tightener AU23",
        # P3: Upper-face
        "strongly lowered and furrowed eyebrows with tense narrowed eyes and an intense stare",
        # P4: Lower-face
        "firmly pressed or tightened lips with visible tension around the mouth and jaw",
        # P5: Combined
        "an angry face with lowered brows, narrowed intense eyes, and tightly pressed lips, consistent with AU4, AU7, and AU23",
    ],
    1: [  # 1: disgust
        # P1: Emotion-level
        "a facial expression of disgust showing strong aversion and distaste",
        # P2: AU-level
        "disgust characterized by nose wrinkler AU9, upper lip raiser AU10, and lip corner depressor AU15",
        # P3: Upper-face
        "wrinkled nose bridge with squinted eyes, lowered brows, and crinkled lower eyelids",
        # P4: Lower-face
        "raised upper lip sneering upward with turned-down mouth corners and a raised chin",
        # P5: Combined
        "a disgusted face with a wrinkled nose, sneering raised upper lip, and downturned mouth corners, consistent with AU9, AU10, and AU15",
    ],
    2: [  # 2: fear
        # P1: Emotion-level
        "a facial expression of fear with a terrified and alarmed look",
        # P2: AU-level
        "fear characterized by inner brow raiser AU1, outer brow raiser AU2, upper lid raiser AU5, and lip stretcher AU20",
        # P3: Upper-face
        "raised and drawn together inner eyebrows with wide open startled eyes displaying white sclera",
        # P4: Lower-face
        "horizontally stretched lips pulled back toward ears with a slightly open tense mouth",
        # P5: Combined
        "a fearful face with high raised brows, wide open frightened eyes, and horizontally stretched lips, consistent with AU1, AU2, AU5, and AU20",
    ],
    3: [  # 3: happy
        # P1: Emotion-level
        "a facial expression of happiness with a joyful and cheerful appearance",
        # P2: AU-level
        "happiness characterized by cheek raiser AU6 and lip corner puller AU12",
        # P3: Upper-face
        "raised cheeks with crinkled eye corners forming visible crow's feet and warm smiling eyes",
        # P4: Lower-face
        "lip corners pulled upward and backward in a broad smile showing visible teeth",
        # P5: Combined
        "a happy face with broad upward smiling lips, raised cheeks, and crinkled smiling eyes, consistent with AU6 and AU12",
    ],
    4: [  # 4: sad
        # P1: Emotion-level
        "a facial expression of sadness with a somber and sorrowful look",
        # P2: AU-level
        "sadness characterized by inner brow raiser AU1, brow lowerer AU4, and lip corner depressor AU15",
        # P3: Upper-face
        "inner corners of the eyebrows raised and slanted upward with drooping upper eyelids and dull eyes",
        # P4: Lower-face
        "downturned lip corners with a trembling or pouty mouth and loose jaw",
        # P5: Combined
        "a sad face with slanted raised inner brows, drooping eyelids, and downturned mouth corners, consistent with AU1, AU4, and AU15",
    ],
    5: [  # 5: surprise
        # P1: Emotion-level
        "a facial expression of surprise with an astonished and amazed appearance",
        # P2: AU-level
        "surprise characterized by inner brow raiser AU1, outer brow raiser AU2, upper lid raiser AU5, and jaw drop AU26",
        # P3: Upper-face
        "high curved eyebrows raised up the forehead with widely opened round un-squinted eyes",
        # P4: Lower-face
        "dropped jaw creating an oval-shaped open mouth with relaxed lips",
        # P5: Combined
        "a surprised face with high arched brows, widely opened staring eyes, and a dropped open jaw, consistent with AU1, AU2, AU5, and AU26",
    ],
    6: [  # 6: neutral
        # P1: Emotion-level
        "a calm and neutral facial expression without emotion",
        # P2: AU-level
        "a neutral face characterized by the absence of active action units",
        # P3: Upper-face
        "relaxed eyebrows and smooth forehead with calm resting eyes",
        # P4: Lower-face
        "closed relaxed lips in a natural resting posture with no tension in the mouth or jaw",
        # P5: Combined
        "a neutral facial expression with balanced resting features, smooth brows, calm eyes, and closed relaxed lips",
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


def compute_prompt_hash(prompts_per_class: Dict[int, List[str]]) -> tuple[str, int]:
    """
    Computes a deterministic SHA256 hash and total count for a prompt dictionary.
    Keys are sorted numerically and prompt lists are serialized predictably.
    """
    sorted_data = {str(k): prompts_per_class[k] for k in sorted(prompts_per_class.keys())}
    json_str = json.dumps(sorted_data, sort_keys=True, ensure_ascii=True)
    prompt_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    prompt_count = sum(len(v) for v in prompts_per_class.values())
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
    prompts_per_class: Optional[Dict[int, List[str]]] = None,
    embedding_dim: int = 512,
    multi_prototype: bool = False,
) -> np.ndarray:
    """
    Computes or loads text prototypes for 7 FER emotion classes using a frozen REAL CLIP text encoder.
    Synthetic/random prototype fallbacks are COMPLETELY DISABLED.
    If loading CLIP fails, raises RuntimeError immediately.
    """
    if prompts_per_class is None:
        prompts_per_class = DEFAULT_AU_EMOTION_PROMPTS

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
                prompts = prompts_per_class[c]
                inputs = tokenizer(prompts, padding=True, return_tensors="pt")
                if hasattr(text_encoder, "get_text_features"):
                    embeds = text_encoder.get_text_features(**inputs)
                elif hasattr(text_encoder, "text_model"):
                    outputs = text_encoder(**inputs)
                    embeds = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[0][:, 0, :]
                else:
                    outputs = text_encoder(**inputs)
                    embeds = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[0][:, 0, :]

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

