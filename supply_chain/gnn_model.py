"""
supply_chain/gnn_model.py
==========================
Spatio-Temporal Graph Attention Network (ST-GAT) for Blast Radius Prediction.

Architecture
------------
  Input projection   : Linear(4  → 64)
  GAT Layer 1        : GATConv(64  → 64,  heads=4, concat=True)   → 256-dim
  GAT Layer 2        : GATConv(256 → 64,  heads=4, concat=True)   → 256-dim
  GAT Layer 3        : GATConv(256 → 64,  heads=1, concat=False)  → 64-dim
  Temporal Attention : MultiheadAttention(64, heads=4)   [blast propagation]
  Risk Head          : MLP → Sigmoid → risk ∈ [0, 1] per node

The three GAT layers model 1-hop, 2-hop, and 3-hop cascading risk within
an 8-hour delivery window.  Edge features (reliability, delay, distance)
are fused at every attention step via GATConv's edge_dim mechanism.

Target parameters: ~150 k  (actual printed at runtime).

In production this model is trained on labelled disruption events.
For the demo, `demo_warm_init()` biases the weights so that
low-health + high-backlog nodes immediately receive elevated risk scores.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn   import GATConv


# ─── ST-GAT Model ────────────────────────────────────────────────────────────

class STGATBlastRadiusModel(nn.Module):
    """
    3-layer Graph Attention Network with temporal self-attention.

    Parameters
    ----------
    node_feature_dim : int  — GNN node feature width  (default 4)
    edge_feature_dim : int  — GNN edge feature width  (default 3)
    hidden_dim       : int  — hidden units per layer   (default 64)
    num_heads        : int  — attention heads          (default 4)
    dropout          : float
    """

    def __init__(
        self,
        node_feature_dim: int   = 4,
        edge_feature_dim: int   = 3,
        hidden_dim:       int   = 64,
        num_heads:        int   = 4,
        dropout:          float = 0.10,
    ) -> None:
        super().__init__()

        H  = hidden_dim
        Nh = num_heads
        Ed = edge_feature_dim

        # ── Input Projection ──────────────────────────────────────────────────
        self.input_proj = nn.Linear(node_feature_dim, H)
        self.input_norm = nn.LayerNorm(H)

        # ── GAT Layer 1: 1-hop neighbour influence ───────────────────────────
        self.gat1 = GATConv(
            in_channels  = H,
            out_channels = H,
            heads        = Nh,
            edge_dim     = Ed,
            dropout      = dropout,
            concat       = True,   # output: H * Nh = 256
            add_self_loops=True,
        )
        self.norm1 = nn.LayerNorm(H * Nh)

        # ── GAT Layer 2: 2-hop blast-radius ──────────────────────────────────
        self.gat2 = GATConv(
            in_channels  = H * Nh,
            out_channels = H,
            heads        = Nh,
            edge_dim     = Ed,
            dropout      = dropout,
            concat       = True,   # output: H * Nh = 256
            add_self_loops=True,
        )
        self.norm2 = nn.LayerNorm(H * Nh)

        # ── GAT Layer 3: 3-hop propagation (8-h window) ──────────────────────
        self.gat3 = GATConv(
            in_channels  = H * Nh,
            out_channels = H,
            heads        = 1,
            edge_dim     = Ed,
            dropout      = dropout,
            concat       = False,  # output: H = 64
            add_self_loops=True,
        )
        self.norm3 = nn.LayerNorm(H)

        # ── Temporal Self-Attention ───────────────────────────────────────────
        # Models how the disruption signal propagates over simulated time steps
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim   = H,
            num_heads   = Nh,
            dropout     = dropout,
            batch_first = True,
        )
        self.temporal_norm = nn.LayerNorm(H)

        # ── Skip Connection (input → output) ─────────────────────────────────
        self.skip_proj = nn.Linear(H, H)

        # ── Risk Prediction Head ──────────────────────────────────────────────
        self.risk_head = nn.Sequential(
            nn.Linear(H, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),         # risk ∈ [0, 1]
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    # ── Weight Initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def demo_warm_init(self) -> None:
        """
        Bias initial weights so that the model immediately produces
        meaningful (non-random) risk scores without training data.

        Heuristic encoded:
          node feature[0] = backlog_norm  → POSITIVE contribution to risk
          node feature[1] = health_status → NEGATIVE contribution to risk
          node feature[2] = volume_norm   → small positive contribution
          node feature[3] = priority      → small contribution
        """
        with torch.no_grad():
            w = self.input_proj.weight.data   # shape [64, 4]
            w.zero_()
            # backlog  → high activation in first 16 channels
            w[:16, 0] =  1.80
            # health   → inverse signal in next 16 channels
            w[16:32, 1] = -1.80
            # volume   → mild signal
            w[32:48, 2] =  0.60
            # priority → mild signal
            w[48:,  3] =  0.40

            # Risk head: bias toward outputting mid-range values
            last_linear = [m for m in self.risk_head.modules()
                           if isinstance(m, nn.Linear)][-1]
            last_linear.bias.data.fill_(-1.5)   # sigmoid(-1.5) ≈ 0.18 baseline

    # ── Forward Pass ──────────────────────────────────────────────────────────

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        data : torch_geometric.data.Data
            data.x         — [N, 4]  node features
            data.edge_index — [2, E] edge connectivity (COO)
            data.edge_attr  — [E, 3] edge features

        Returns
        -------
        risk_scores : Tensor [N]       — per-node risk ∈ [0, 1]
        attn_weights: Tensor [1, N, N] — temporal attention matrix
        """
        x, ei, ea = data.x, data.edge_index, data.edge_attr

        # ── Input projection ──────────────────────────────────────────────────
        h = F.gelu(self.input_norm(self.input_proj(x)))   # [N, H]
        h_skip = self.skip_proj(h)                         # [N, H]  residual

        # ── GAT Layer 1 — direct neighbourhood ───────────────────────────────
        h1 = self.gat1(h, ei, ea)          # [N, H*Nh]
        h1 = F.elu(self.norm1(h1))
        h1 = self.dropout(h1)

        # ── GAT Layer 2 — 2-hop blast radius ─────────────────────────────────
        h2 = self.gat2(h1, ei, ea)         # [N, H*Nh]
        h2 = F.elu(self.norm2(h2))
        h2 = self.dropout(h2)

        # ── GAT Layer 3 — 3-hop propagation (~8 h window) ────────────────────
        h3 = self.gat3(h2, ei, ea)         # [N, H]
        h3 = F.elu(self.norm3(h3))

        # ── Temporal Self-Attention ───────────────────────────────────────────
        # Treat all nodes as a "sequence" and let attention model cross-node
        # blast propagation patterns learned during training.
        h_seq   = h3.unsqueeze(0)                               # [1, N, H]
        h_t, attn_w = self.temporal_attn(h_seq, h_seq, h_seq)  # [1, N, H]
        h_t     = h_t.squeeze(0)                                # [N, H]
        h_out   = self.temporal_norm(h_t + h_skip)             # residual

        # ── Risk score ────────────────────────────────────────────────────────
        risk = self.risk_head(h_out).squeeze(-1)   # [N]

        return risk, attn_w

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        """Per-module parameter counts."""
        breakdown: Dict[str, int] = {}
        for name, mod in self.named_children():
            cnt = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            breakdown[name] = cnt
        return breakdown

    # ── Graph Data Builder ────────────────────────────────────────────────────

    @staticmethod
    def build_graph_data(
        env_state:  dict,
        node_list:  List[str],
        edge_index: List[List[int]],
        edge_attr:  List[List[float]],
    ) -> Data:
        """
        Convert raw environment state into a `torch_geometric.data.Data` object.

        Node feature vector (4-dim, normalised):
          [backlog_norm, health_status, volume_norm, priority_index]

        Edge feature vector (3-dim):
          [route_reliability, delay_norm, distance_weight]
        """
        node_features: List[List[float]] = []
        for nid in node_list:
            node = env_state["nodes"][nid]
            node_features.append(node.to_feature_vec())

        x = torch.tensor(node_features, dtype=torch.float32)

        if edge_index:
            ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
            ea = torch.tensor(edge_attr,  dtype=torch.float32)                 # [E, 3]
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros((0, 3), dtype=torch.float32)

        return Data(x=x, edge_index=ei, edge_attr=ea)


# ─── Heuristic Blast-Radius Propagator ───────────────────────────────────────

def propagate_blast_radius(
    env_state:  dict,
    node_list:  List[str],
    gnn_output: torch.Tensor,
    num_hops:   int = 3,
    blend_alpha: float = 0.45,
) -> Dict[str, float]:
    """
    Blend GNN output with graph-topology propagation for demo clarity.

    For production use, remove blending and rely solely on the trained GNN.

    Algorithm
    ---------
    1. Compute per-node heuristic score from features.
    2. Propagate heuristic scores downstream through the supply graph
       (attenuated by route reliability each hop).
    3. Blend: final = alpha * gnn + (1 - alpha) * propagated_heuristic.

    Parameters
    ----------
    blend_alpha : float  — weight of GNN output vs heuristic (0 = pure heuristic)
    """
    nodes = env_state["nodes"]
    edges = env_state["edges"]

    # Step 1: Heuristic base scores
    heuristic: Dict[str, float] = {}
    for nid, node in nodes.items():
        backlog_norm  = min(node.current_backlog / 1000.0, 1.0)
        health_factor = 1.0 - node.health_status
        failure_spike = 0.55 if node.failure_injected else 0.0
        heuristic[nid] = min(
            1.0,
            health_factor * 0.45 + backlog_norm * 0.30 + failure_spike,
        )

    # Step 2: Downstream propagation (simulate blast-radius cascades)
    propagated = heuristic.copy()
    for _ in range(num_hops):
        updated = propagated.copy()
        for edge in edges:
            src, tgt = edge.source, edge.target
            if src in propagated and tgt in propagated:
                cascade = propagated[src] * edge.route_reliability * 0.68
                # Only raise risk, never lower it (monotone propagation)
                if cascade > updated[tgt]:
                    updated[tgt] = min(1.0, cascade)
        propagated = updated

    # Step 3: Blend with GNN output
    final: Dict[str, float] = {}
    gnn_arr = gnn_output.detach().cpu().numpy()
    for i, nid in enumerate(node_list):
        gnn_score = float(gnn_arr[i]) if i < len(gnn_arr) else 0.5
        final[nid] = blend_alpha * gnn_score + (1.0 - blend_alpha) * propagated.get(nid, 0.5)

    return final
