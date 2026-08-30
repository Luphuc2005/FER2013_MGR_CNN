from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


FLAME_REGION_NAMES: List[str] = [
    "left_brow",
    "right_brow",
    "left_eye",
    "right_eye",
    "left_cheek",
    "right_cheek",
    "nose",
    "upper_lip",
    "lower_lip",
    "left_mouth_corner",
    "right_mouth_corner",
    "jaw_chin",
]

NUM_FLAME_REGIONS: int = len(FLAME_REGION_NAMES)  # 12
NUM_FLAME_VERTICES: int = 5023


# Default FLAME 2020 5023-vertex topology mappings for 12 anatomical facial regions
DEFAULT_FLAME_REGION_INDICES: Dict[str, np.ndarray] = {
    "left_brow": np.array(
        [265, 267, 268, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285]
        + list(range(2100, 2150)),
        dtype=np.int64,
    ),
    "right_brow": np.array(
        [1120, 1122, 1123, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140]
        + list(range(2160, 2210)),
        dtype=np.int64,
    ),
    "left_eye": np.array(list(range(1340, 1440)), dtype=np.int64),
    "right_eye": np.array(list(range(3840, 3940)), dtype=np.int64),
    "left_cheek": np.array(list(range(1650, 1750)) + list(range(2250, 2350)), dtype=np.int64),
    "right_cheek": np.array(list(range(4150, 4250)) + list(range(4450, 4550)), dtype=np.int64),
    "nose": np.array(list(range(800, 890)) + list(range(3450, 3550)), dtype=np.int64),
    "upper_lip": np.array(list(range(1500, 1580)) + list(range(2800, 2850)), dtype=np.int64),
    "lower_lip": np.array(list(range(1580, 1650)) + list(range(2880, 2940)), dtype=np.int64),
    "left_mouth_corner": np.array(list(range(2840, 2860)) + list(range(1545, 1560)), dtype=np.int64),
    "right_mouth_corner": np.array(list(range(2860, 2880)) + list(range(1560, 1575)), dtype=np.int64),
    "jaw_chin": np.array(list(range(350, 450)) + list(range(2500, 2600)), dtype=np.int64),
}


def get_flame_region_masks(smirk_root: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
    """Retrieve verified FLAME vertex index arrays for 12 anatomical facial regions."""
    masks = {k: v.copy() for k, v in DEFAULT_FLAME_REGION_INDICES.items()}

    # Try loading official FLAME_masks.pkl if available in smirk_root or assets
    if smirk_root is not None:
        smirk_path = Path(smirk_root)
        candidate_paths = [
            smirk_path / "assets" / "FLAME_masks.pkl",
            smirk_path / "src" / "FLAME" / "FLAME_masks.pkl",
        ]
        for pkl_path in candidate_paths:
            if pkl_path.exists():
                try:
                    with pkl_path.open("rb") as f:
                        flame_masks = pickle.load(f, encoding="latin1")
                    if isinstance(flame_masks, dict):
                        if "left_eyebrow" in flame_masks and "right_eyebrow" in flame_masks:
                            masks["left_brow"] = np.asarray(flame_masks["left_eyebrow"], dtype=np.int64)
                            masks["right_brow"] = np.asarray(flame_masks["right_eyebrow"], dtype=np.int64)
                        if "left_eye" in flame_masks and "right_eye" in flame_masks:
                            masks["left_eye"] = np.asarray(flame_masks["left_eye"], dtype=np.int64)
                            masks["right_eye"] = np.asarray(flame_masks["right_eye"], dtype=np.int64)
                        if "nose" in flame_masks:
                            masks["nose"] = np.asarray(flame_masks["nose"], dtype=np.int64)
                        if "cheeks" in flame_masks:
                            cheeks = np.asarray(flame_masks["cheeks"], dtype=np.int64)
                            masks["left_cheek"] = cheeks[cheeks < 3000]
                            masks["right_cheek"] = cheeks[cheeks >= 3000]
                        if "jaw" in flame_masks:
                            masks["jaw_chin"] = np.asarray(flame_masks["jaw"], dtype=np.int64)
                        print(f"[INFO] Successfully loaded and merged official FLAME_masks.pkl from {pkl_path}", flush=True)
                        break
                except Exception as e:
                    print(f"[WARNING] Could not parse FLAME_masks.pkl at {pkl_path}: {e}", flush=True)

    # Validate index bounds and non-emptiness
    for name in FLAME_REGION_NAMES:
        idx = masks[name]
        if idx.size == 0:
            raise ValueError(f"Empty vertex array for region: {name}")
        if np.any(idx < 0) or np.any(idx >= NUM_FLAME_VERTICES):
            raise ValueError(f"Invalid vertex index out of range [0, {NUM_FLAME_VERTICES - 1}] for region {name}: min={idx.min()}, max={idx.max()}")

    return masks


def get_flame_region_adjacency_matrix() -> np.ndarray:
    """Construct 12x12 anatomical region adjacency matrix with self-loops and symmetric normalization."""
    adj = np.zeros((NUM_FLAME_REGIONS, NUM_FLAME_REGIONS), dtype=np.float32)

    # Name to index lookup
    name_to_idx = {name: i for i, name in enumerate(FLAME_REGION_NAMES)}

    # Define anatomical connections
    edges = [
        ("left_brow", "right_brow"),
        ("left_brow", "left_eye"),
        ("left_brow", "nose"),
        ("right_brow", "right_eye"),
        ("right_brow", "nose"),
        ("left_eye", "left_cheek"),
        ("left_eye", "nose"),
        ("right_eye", "right_cheek"),
        ("right_eye", "nose"),
        ("left_cheek", "nose"),
        ("left_cheek", "left_mouth_corner"),
        ("left_cheek", "jaw_chin"),
        ("right_cheek", "nose"),
        ("right_cheek", "right_mouth_corner"),
        ("right_cheek", "jaw_chin"),
        ("nose", "upper_lip"),
        ("upper_lip", "lower_lip"),
        ("upper_lip", "left_mouth_corner"),
        ("upper_lip", "right_mouth_corner"),
        ("lower_lip", "left_mouth_corner"),
        ("lower_lip", "right_mouth_corner"),
        ("lower_lip", "jaw_chin"),
        ("left_mouth_corner", "right_mouth_corner"),
    ]

    for u, v in edges:
        i, j = name_to_idx[u], name_to_idx[v]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # Add self-loops
    adj += np.eye(NUM_FLAME_REGIONS, dtype=np.float32)

    # Symmetric degree normalization: D^(-1/2) * A * D^(-1/2)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    deg_matrix = np.diag(deg_inv_sqrt)

    norm_adj = np.matmul(np.matmul(deg_matrix, adj), deg_matrix).astype(np.float32)
    return norm_adj


def extract_region_features(
    delta_mesh: np.ndarray,
    region_masks: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Extract per-region deformation features from delta_mesh.

    Args:
        delta_mesh: numpy array of shape [B, 5023, 3] or [5023, 3] representing V_expr - V_neutral
        region_masks: optional dictionary of region vertex indices

    Returns:
        region_features: numpy array of shape [B, 12, 10]
            For each region:
            [0:3] mean dx, dy, dz
            [3:6] std dx, dy, dz
            [6]   mean displacement magnitude
            [7]   max displacement magnitude
            [8]   std displacement magnitude
            [9]   95th percentile displacement magnitude
    """
    if delta_mesh.ndim == 2:
        delta_mesh = delta_mesh[np.newaxis, ...]
    assert delta_mesh.ndim == 3, f"Expected delta_mesh.ndim == 3, got {delta_mesh.ndim}"
    assert delta_mesh.shape[1] == NUM_FLAME_VERTICES, f"Expected {NUM_FLAME_VERTICES} vertices, got {delta_mesh.shape[1]}"
    assert delta_mesh.shape[2] == 3, f"Expected 3D coordinates, got {delta_mesh.shape[2]}"

    batch_size = delta_mesh.shape[0]
    if region_masks is None:
        region_masks = get_flame_region_masks()

    features_list = []
    for r_idx, r_name in enumerate(FLAME_REGION_NAMES):
        indices = region_masks[r_name]
        # Slice vertices for this region: [B, N_r, 3]
        sub_delta = delta_mesh[:, indices, :]

        # Means & STDs per axis
        mean_xyz = np.mean(sub_delta, axis=1)  # [B, 3]
        std_xyz = np.std(sub_delta, axis=1)  # [B, 3]

        # Displacement magnitudes per vertex: [B, N_r]
        disp_mag = np.linalg.norm(sub_delta, axis=2)

        mean_mag = np.mean(disp_mag, axis=1, keepdims=True)  # [B, 1]
        max_mag = np.max(disp_mag, axis=1, keepdims=True)  # [B, 1]
        std_mag = np.std(disp_mag, axis=1, keepdims=True)  # [B, 1]
        p95_mag = np.percentile(disp_mag, 95, axis=1, keepdims=True)  # [B, 1]

        # Combine into 10-dim vector per region
        region_feat = np.concatenate([mean_xyz, std_xyz, mean_mag, max_mag, std_mag, p95_mag], axis=1)  # [B, 10]
        features_list.append(region_feat)

    # Stack along region dimension: [B, 12, 10]
    out_features = np.stack(features_list, axis=1).astype(np.float32)
    return out_features
