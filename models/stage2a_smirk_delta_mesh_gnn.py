from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from utils.flame_region_mapping import get_flame_region_adjacency_matrix


class GCNLayer(tf.keras.layers.Layer):
    """Graph Convolutional Layer using fixed normalized adjacency matrix A_hat."""

    def __init__(self, output_dim: int, adj_matrix: np.ndarray, dropout_rate: float = 0.2, name: str = "gcn_layer"):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.adj = tf.constant(adj_matrix, dtype=tf.float32)  # [12, 12]
        self.dropout_rate = dropout_rate

        self.dense = tf.keras.layers.Dense(output_dim, name="gcn_weight")
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="gcn_ln")
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name="gcn_dropout")

    def call(self, x, training=False):
        # x: [B, N, D_in], adj: [N, N]
        # Message passing: A_hat * X
        # Batch matmul: einsum or reshape
        aggr = tf.einsum("ij,bjk->bik", self.adj, x)  # [B, N, D_in]
        out = self.dense(aggr)  # [B, N, D_out]
        out = self.ln(out + x if x.shape[-1] == self.output_dim else out)
        out = tf.nn.gelu(out)
        out = self.dropout(out, training=training)
        return out


class Stage2ASMIRKDeltaMeshGNN(tf.keras.Model):
    """Stage 2A: 3D-Only FER Probe using FLAME Delta Mesh Region Graph.

    Pipeline:
        Region Features [B, 12, 10] -> Node Linear (128) -> 2-layer GCN (128) ->
        Global Region Attention Pooling (256) -> Dense (512) -> Dense (7 Logits).

    Zero RGB input. 100% 3D Deformation Geometry.
    """

    def __init__(self, cfg: Dict):
        super().__init__(name=cfg.get("model", {}).get("name", "stage2a_smirk_delta_mesh_gnn"))
        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})

        self.num_classes = int(data_cfg.get("num_classes", 7))
        self.num_regions = int(model_cfg.get("num_regions", 12))
        self.region_feature_dim = int(model_cfg.get("region_feature_dim", 10))
        self.node_proj_dim = int(model_cfg.get("node_proj_dim", 128))
        self.gcn_hidden_dim = int(model_cfg.get("gcn_hidden_dim", 128))
        self.fc_geometry_dim = int(model_cfg.get("fc_geometry_dim", 512))
        self.dropout_rate = float(model_cfg.get("dropout", 0.20))
        self._shape_logged = False

        # Load normalized facial region graph adjacency [12, 12]
        self.adj_matrix = get_flame_region_adjacency_matrix()

        # Node Feature Projection
        self.node_project = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name="node_ln"),
                tf.keras.layers.Dense(self.node_proj_dim, name="node_dense"),
                tf.keras.layers.Activation(tf.nn.gelu, name="node_gelu"),
                tf.keras.layers.Dropout(self.dropout_rate, name="node_dropout"),
            ],
            name="node_feature_projection",
        )

        # 2-Layer Graph Convolution Network
        self.gcn_1 = GCNLayer(self.gcn_hidden_dim, self.adj_matrix, dropout_rate=self.dropout_rate, name="gcn_layer_1")
        self.gcn_2 = GCNLayer(self.gcn_hidden_dim, self.adj_matrix, dropout_rate=self.dropout_rate, name="gcn_layer_2")

        # Region Attention Pooling
        self.region_attn_dense = tf.keras.layers.Dense(1, name="region_attention_weight")

        # Dense Geometry Feature Encoder (-> 512)
        self.geometry_feature_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6, name="geo_ln"),
                tf.keras.layers.Dense(self.fc_geometry_dim, activation=tf.nn.gelu, name="geo_dense"),
                tf.keras.layers.Dropout(self.dropout_rate, name="geo_dropout"),
            ],
            name="dense_geometry_feature_encoder",
        )

        # 3D-Only FER Classifier Head (-> 7)
        self.classifier = tf.keras.layers.Dense(self.num_classes, name="fer_3d_logits")

    def call(self, inputs, training=False):
        # inputs can be dict or tensor [B, 12, 10]
        region_features = inputs["region_features"] if isinstance(inputs, dict) else inputs
        region_features = tf.cast(region_features, tf.float32)

        # 1. Project region features: [B, 12, 10] -> [B, 12, 128]
        h0 = self.node_project(region_features, training=training)

        # 2. 2-layer GCN message passing
        h1 = self.gcn_1(h0, training=training)  # [B, 12, 128]
        h2 = self.gcn_2(h1, training=training)  # [B, 12, 128]

        # 3. Global Region Attention & Mean Pooling
        attn_logits = self.region_attn_dense(h2)  # [B, 12, 1]
        attn_weights = tf.nn.softmax(attn_logits, axis=1)  # [B, 12, 1]
        g_attn = tf.reduce_sum(h2 * attn_weights, axis=1)  # [B, 128]
        g_mean = tf.reduce_mean(h2, axis=1)  # [B, 128]
        g_combined = tf.concat([g_attn, g_mean], axis=-1)  # [B, 256]

        # 4. Dense 512-d Geometry Feature
        geo_feature_512 = self.geometry_feature_mlp(g_combined, training=training)  # [B, 512]

        # 5. 3D Classifier Logits
        logits = tf.cast(self.classifier(geo_feature_512), tf.float32)  # [B, 7]

        if not self._shape_logged:
            self._shape_logged = True
            print("[Stage2ASMIRKDeltaMeshGNN] Tensor Shape Trace:", flush=True)
            print(f"  region_features input: {region_features.shape}", flush=True)
            print(f"  node_project (h0): {h0.shape}", flush=True)
            print(f"  gcn_layer_1 (h1): {h1.shape}", flush=True)
            print(f"  gcn_layer_2 (h2): {h2.shape}", flush=True)
            print(f"  attn_weights: {attn_weights.shape}", flush=True)
            print(f"  graph_pooling (g_combined): {g_combined.shape}", flush=True)
            print(f"  geo_feature_512: {geo_feature_512.shape}", flush=True)
            print(f"  3d_fer_logits: {logits.shape}", flush=True)

        return {
            "logits": logits,
            "geo_feature_512": geo_feature_512,
            "region_attention": attn_weights,
        }
