from __future__ import annotations

"""
train_gnn.py
============
Synthetic training loop for the ST-GAT blast-radius model.

This script generates supervised data from the logistics simulator by using
the existing heuristic blast-radius propagator as a "teacher" signal and
trains the GNN to approximate those risk scores.

Run once from the project root:

    python train_gnn.py

It will produce `trained_gnn_weights.pt` in the current directory, which
`initialise_gnn_weights` in `gnn_model.py` will load when available.
"""

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.optim import Adam

from supply_chain.environment import LogisticsEnvironment
from supply_chain.gnn_model import (
    STGATBlastRadiusModel,
    propagate_blast_radius,
)


def generate_batch(
    env: LogisticsEnvironment,
    model: STGATBlastRadiusModel,
    blend_alpha: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Generate a single training example by advancing the environment one tick,
    building a graph, and computing a heuristic "teacher" risk label using
    the existing blast-radius propagator (with blend_alpha=0 for pure heuristic).
    """
    env_state = env.tick()
    node_list, edge_index, edge_attr = env.get_pyg_tensors()
    graph_data = STGATBlastRadiusModel.build_graph_data(
        env_state, node_list, edge_index, edge_attr
    )

    # Use zero GNN output and rely purely on heuristic propagation for labels.
    with torch.no_grad():
        dummy_gnn = torch.zeros(len(node_list), dtype=torch.float32)
        teacher_scores = propagate_blast_radius(
            env_state=env_state,
            node_list=node_list,
            gnn_output=dummy_gnn,
            num_hops=3,
            blend_alpha=blend_alpha,  # typically 0.0 for pure heuristic
        )

    # Convert to tensors aligned with node_list ordering
    y = torch.tensor(
        [teacher_scores[nid] for nid in node_list],
        dtype=torch.float32,
    )
    return {"graph": graph_data, "target": y}


def train_synthetic(
    epochs: int = 3,
    steps_per_epoch: int = 300,
    lr: float = 3e-4,
    weight_path: str = "trained_gnn_weights.pt",
) -> None:
    torch.manual_seed(42)

    env = LogisticsEnvironment(seed=42)
    model = STGATBlastRadiusModel(
        node_feature_dim=4,
        edge_feature_dim=3,
        hidden_dim=64,
        num_heads=4,
        dropout=0.10,
    )

    model.train()
    opt = Adam(model.parameters(), lr=lr)

    total_steps = epochs * steps_per_epoch
    step = 0

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            batch = generate_batch(env, model, blend_alpha=0.0)
            graph = batch["graph"]
            target = batch["target"]

            opt.zero_grad()
            pred, _ = model(graph)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()

            step += 1
            running_loss += float(loss.item())

        avg_loss = running_loss / steps_per_epoch
        print(f"[Epoch {epoch}/{epochs}] avg MSE loss: {avg_loss:.4f}")

    # Save trained weights
    torch.save(model.state_dict(), weight_path)
    print(f"Saved trained GNN weights to {weight_path}")


if __name__ == "__main__":
    train_synthetic()

