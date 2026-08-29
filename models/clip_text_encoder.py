from __future__ import annotations

import os
from typing import Dict, List, Optional
import numpy as np

# AU-aware Action Unit text description templates for 7 FER emotions
DEFAULT_AU_EMOTION_PROMPTS: Dict[int, List[str]] = {
    0: [  # Angry
        "a face showing anger with lowered eyebrows, tightened eyelids, and pressed lips (AU4, AU7, AU23)",
        "a person expressing anger with furrowed brows, intense glare, and firmly closed mouth",
        "a facial expression of anger with brow lowerer, upper lid raiser, and lip tightener",
        "a close-up portrait of an angry person with tense facial muscles and hostile expression",
        "an angry facial expression with lowered brow, glare, and tight lips",
    ],
    1: [  # Disgust
        "a face showing disgust with wrinkled nose, raised upper lip, and narrowed eyes (AU9, AU10)",
        "a person expressing disgust with nose wrinkler, lip corner depressor, and cheek raiser",
        "a facial expression of disgust with sneering nose and turned-up upper lip",
        "a close-up portrait of a disgusted person displaying aversion and wrinkled nose",
        "a disgusted facial expression with curled upper lip and crinkled nose bridge",
    ],
    2: [  # Fear
        "a face showing fear with raised inner and outer eyebrows, widened eyes, and parted lips (AU1, AU2, AU5, AU20)",
        "a person expressing fear with wide eyes, open mouth, and stretched lip corners",
        "a facial expression of fear with eyebrow elevator, upper lid raiser, and mouth stretcher",
        "a close-up portrait of a fearful person with terrified gaze and gasping mouth",
        "a fearful facial expression with wide open eyes and tense raised brows",
    ],
    3: [  # Happy
        "a face showing happiness with raised cheeks and pulled lip corners in a wide smile (AU6, AU12)",
        "a person expressing happiness with a cheerful smile, crow's feet around eyes, and raised cheeks",
        "a facial expression of joy and happiness with cheek raiser and lip corner puller",
        "a close-up portrait of a happy smiling person with radiant joyful eyes",
        "a happy facial expression with broad smile, visible teeth, and crinkled eye corners",
    ],
    4: [  # Sad
        "a face showing sadness with drooping eyelids, inner eyebrows pulled up, and downturned mouth corners (AU1, AU4, AU15)",
        "a person expressing sadness with sorrowful eyes, inner brow raiser, and lip corner depressor",
        "a facial expression of sadness with tearful gaze, raised inner eyebrows, and frowning mouth",
        "a close-up portrait of a sad person with gloomy melancholic expression",
        "a sad facial expression with downturned lip corners and raised inner brows",
    ],
    5: [  # Surprise
        "a face showing surprise with high raised eyebrows, widely open eyes, and dropped jaw (AU1, AU2, AU5, AU26)",
        "a person expressing surprise with elevated brows, open mouth, and shocked expression",
        "a facial expression of surprise with inner and outer brow elevator, eye widener, and jaw drop",
        "a close-up portrait of a surprised person with astonished wide gaze and open jaw",
        "a surprised facial expression with arched brows, round open eyes, and dropped mouth",
    ],
    6: [  # Neutral
        "a face showing neutral expression with relaxed facial muscles and calm eyes",
        "a person expressing a neutral state without active action units or facial tension",
        "a facial expression of neutral composure with balanced brows and closed lips",
        "a close-up portrait of a neutral person with a calm, emotionless countenance",
        "a neutral facial expression with tranquil posture and relaxed facial features",
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
    if cache_path is None or (multi_prototype and cache_path == "pretrained/clip_text_prototypes_7emotions.npy"):
        cache_path = "pretrained/clip_text_prototypes_7emotions_multi5.npy" if multi_prototype else "pretrained/clip_text_prototypes_7emotions.npy"

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
