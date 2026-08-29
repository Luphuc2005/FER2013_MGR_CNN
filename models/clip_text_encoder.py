from __future__ import annotations

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


def get_or_compute_clip_text_prototypes(
    model_name: str = "openai/clip-vit-base-patch32",
    cache_path: Optional[str] = None,
    prompts_per_class: Optional[Dict[int, List[str]]] = None,
    embedding_dim: int = 512,
    multi_prototype: bool = False,
) -> np.ndarray:
    """
    Computes or loads text prototypes for 7 FER emotion classes using a frozen CLIP text encoder.
    If multi_prototype is False: returns shape (7, embedding_dim) (averaged across prompts per class).
    If multi_prototype is True: returns shape (7, 5, embedding_dim) (5 individual L2-normalized prompt vectors per class).
    """
    if cache_path is None or (multi_prototype and "clip_text_prototypes_7emotions.npy" in cache_path):
        cache_path = "pretrained/clip_text_prototypes_7emotions_multigranularity_multi5.npy" if multi_prototype else "pretrained/clip_text_prototypes_7emotions.npy"

    print("[CLIP_Text_Encoder] Emotion prototype index alignment trace:", flush=True)
    for class_idx, class_name in EMOTION_CLASS_MAP.items():
        print(f"[CLIP_Text_Encoder]   class_index {class_idx} -> class_name '{class_name}' -> prototype_index [{class_idx}]", flush=True)

    expected_shape = (7, 5, embedding_dim) if multi_prototype else (7, embedding_dim)
    if cache_path and os.path.exists(cache_path):
        try:
            prototypes = np.load(cache_path)
            if prototypes.shape == expected_shape:
                print(f"[CLIP_Text_Encoder] Loaded cached text prototypes from {cache_path} (shape: {prototypes.shape})", flush=True)
                return prototypes.astype(np.float32)
        except Exception as err:
            print(f"[CLIP_Text_Encoder] Warning: Failed to load {cache_path}: {err}. Recomputing...", flush=True)

    if prompts_per_class is None:
        prompts_per_class = DEFAULT_AU_EMOTION_PROMPTS

    # Attempt HuggingFace Transformers (PyTorch or TensorFlow)
    prototypes = _encode_with_transformers(model_name, prompts_per_class, embedding_dim, multi_prototype=multi_prototype)
    
    if prototypes is None:
        print("[CLIP_Text_Encoder] Warning: Could not initialize HuggingFace CLIP model. Falling back to synthetic normalized reference prototypes.", flush=True)
        prototypes = _generate_fallback_prototypes(embedding_dim, multi_prototype=multi_prototype)

    # Save to disk for fast caching
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            np.save(cache_path, prototypes)
            print(f"[CLIP_Text_Encoder] Cached text prototypes saved to {cache_path}", flush=True)
        except Exception as e:
            print(f"[CLIP_Text_Encoder] Warning: Could not save text prototypes to {cache_path}: {e}", flush=True)

    return prototypes.astype(np.float32)


def _encode_with_transformers(
    model_name: str,
    prompts_per_class: Dict[int, List[str]],
    embedding_dim: int,
    multi_prototype: bool = False,
) -> Optional[np.ndarray]:
    """Encodes text prompts using HuggingFace transformers CLIP text model (PyTorch or TF)."""
    # 1. Try PyTorch Transformers
    try:
        import torch
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        print(f"[CLIP_Text_Encoder] Encoding AU-aware prompts using PyTorch '{model_name}' (multi_prototype={multi_prototype})...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(model_name)
        text_encoder.eval()

        class_prototypes = []
        with torch.no_grad():
            for c in range(7):
                prompts = prompts_per_class[c]
                inputs = tokenizer(prompts, padding=True, return_tensors="pt")
                outputs = text_encoder(**inputs)
                # text_embeds shape: [num_prompts, 512]
                embeds = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[0][:, 0, :]
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                if multi_prototype:
                    class_prototypes.append(embeds.cpu().numpy())
                else:
                    mean_embed = embeds.mean(dim=0)
                    mean_embed = mean_embed / mean_embed.norm(p=2, dim=-1, keepdim=True)
                    class_prototypes.append(mean_embed.cpu().numpy())

        res = np.stack(class_prototypes, axis=0)
        print(f"[CLIP_Text_Encoder] Successfully generated PyTorch CLIP prototypes shape: {res.shape}", flush=True)
        return res
    except Exception as e_pt:
        print(f"[CLIP_Text_Encoder] PyTorch encoding unavailable or failed: {e_pt}", flush=True)

    # 2. Try TensorFlow Transformers
    try:
        import tensorflow as tf
        from transformers import AutoTokenizer, TFCLIPTextModelWithProjection

        print(f"[CLIP_Text_Encoder] Encoding AU-aware prompts using TensorFlow '{model_name}' (multi_prototype={multi_prototype})...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder = TFCLIPTextModelWithProjection.from_pretrained(model_name)

        class_prototypes = []
        for c in range(7):
            prompts = prompts_per_class[c]
            inputs = tokenizer(prompts, padding=True, return_tensors="tf")
            outputs = text_encoder(**inputs)
            embeds = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[0][:, 0, :]
            embeds = embeds / tf.norm(embeds, ord=2, axis=-1, keepdims=True)
            if multi_prototype:
                class_prototypes.append(embeds.numpy())
            else:
                mean_embed = tf.reduce_mean(embeds, axis=0)
                mean_embed = mean_embed / tf.norm(mean_embed, ord=2, axis=-1, keepdims=True)
                class_prototypes.append(mean_embed.numpy())

        res = np.stack(class_prototypes, axis=0)
        print(f"[CLIP_Text_Encoder] Successfully generated TensorFlow CLIP prototypes shape: {res.shape}", flush=True)
        return res
    except Exception as e_tf:
        print(f"[CLIP_Text_Encoder] TensorFlow encoding unavailable or failed: {e_tf}", flush=True)

    return None


def _generate_fallback_prototypes(embedding_dim: int = 512, multi_prototype: bool = False) -> np.ndarray:
    """Generates reproducible L2-normalized reference vectors for 7 classes."""
    rng = np.random.RandomState(42)
    if multi_prototype:
        raw = rng.randn(7, 5, embedding_dim).astype(np.float32)
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        return raw / norms
    else:
        raw = rng.randn(7, embedding_dim).astype(np.float32)
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        return raw / norms
