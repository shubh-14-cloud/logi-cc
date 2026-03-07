#!/usr/bin/env python3
"""
main.py — Agentic Supply Chain Layer
======================================
Full execution loop integrating:
  • LogisticsEnvironment  (discrete-event simulation)
  • STGATBlastRadiusModel (3-layer GNN risk prediction)
  • SupplyChainAgent      (ReAct reasoning + tool-calling)
  • GuardrailLayer        (human-in-the-loop safety)
  • SupplyChainDashboard  (Rich CLI live dashboard)

Usage
-----
  python main.py                              # defaults: 20 ticks, HUB-NYC fails at tick 3
  python main.py --fail-node HUB-CHI         # inject failure at Chicago
  python main.py --ticks 30 --delay 1.5      # 30 ticks, 1.5 s between frames
  python main.py --auto-approve              # demo mode (auto-approves guardrail escalations)
  python main.py --fail-tick 5 --severity 0.9

Architecture Flow (each tick)
------------------------------
  SimPy tick → env.tick()
      ↓
  env.get_pyg_tensors() → build Data object
      ↓
  model.forward(data)  → raw_risk_tensor
      ↓
  propagate_blast_radius() → final risk_scores dict
      ↓
  agent.run_cycle(risk_scores, env_state)
    ├─ observe()   → observation string
    ├─ reason()    → chain-of-thought + ThoughtStep
    └─ act()       → guardrail.intercept() → tool call
      ↓
  dashboard.render()  →  CLI refresh
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Dict

import numpy as np
import torch

# ── Supply chain package imports ──────────────────────────────────────────────
from supply_chain.environment import LogisticsEnvironment
from supply_chain.gnn_model   import STGATBlastRadiusModel, propagate_blast_radius
from supply_chain.agent       import SupplyChainAgent
from supply_chain.guardrails  import GuardrailLayer
from supply_chain.dashboard   import SupplyChainDashboard


# ─── CLI Arguments ────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agentic Supply Chain Layer — ST-GAT Blast Radius Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticks",        type=int,   default=20,
                   help="Total simulation ticks (1 tick = 1 simulated hour)")
    p.add_argument("--fail-node",    type=str,   default="HUB-NYC",
                   help="Node ID where the failure is injected")
    p.add_argument("--fail-tick",    type=int,   default=3,
                   help="Tick at which the failure is injected")
    p.add_argument("--severity",     type=float, default=0.80,
                   help="Failure severity [0.0 – 1.0]")
    p.add_argument("--delay",        type=float, default=2.0,
                   help="Seconds to pause between ticks (set 0 for max speed)")
    p.add_argument("--auto-approve", action="store_true",
                   help="Auto-approve guardrail escalations (demo mode)")
    p.add_argument("--resolve-tick", type=int,   default=None,
                   help="Tick at which to resolve the injected failure (optional)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--hidden-dim",   type=int,   default=64,
                   help="GNN hidden dimension (target ~150 k params)")
    p.add_argument("--heads",        type=int,   default=4,
                   help="GAT attention heads")
    p.add_argument("--blend",        type=float, default=0.45,
                   help="GNN vs heuristic blend (0=pure heuristic, 1=pure GNN)")
    return p


# ─── Banner ───────────────────────────────────────────────────────────────────

_BANNER = (
    "\n"
    "=" * 80 + "\n"
    "   AGENTIC SUPPLY CHAIN LAYER\n"
    "   ST-GAT Blast Radius Prediction  |  ReAct Autonomous Agent\n"
    "   Human-in-the-Loop Guardrails    |  Rich CLI Dashboard\n"
    + "=" * 80 + "\n"
)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_arg_parser().parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(_BANNER)

    # ── Component Initialisation ──────────────────────────────────────────────
    print("  [1/5] Initialising logistics environment ...")
    env = LogisticsEnvironment(seed=args.seed)

    print("  [2/5] Building ST-GAT model ...")
    model = STGATBlastRadiusModel(
        node_feature_dim = 4,
        edge_feature_dim = 3,
        hidden_dim       = args.hidden_dim,
        num_heads        = args.heads,
        dropout          = 0.10,
    )
    model.eval()
    model.demo_warm_init()   # bias weights for meaningful demo risk scores

    param_count  = model.count_parameters()
    param_detail = model.parameter_breakdown()
    print(f"        Total parameters : {param_count:,}")
    for name, cnt in param_detail.items():
        print(f"          {name:<22s}: {cnt:,}")

    print("  [3/5] Configuring guardrail layer ...")
    total_volume = sum(n.incoming_volume for n in env.nodes.values())
    guardrail = GuardrailLayer(
        cost_threshold_usd   = 500.0,
        volume_threshold_pct = 0.10,
        total_network_volume = total_volume,
        auto_approve         = args.auto_approve,
    )
    print(f"        Cost gate    : ${guardrail.cost_threshold:.0f}  |  "
          f"Volume gate: {guardrail.volume_threshold:.0%}  |  "
          f"Mode: {'AUTO-APPROVE (demo)' if args.auto_approve else 'STRICT (human required)'}")

    print("  [4/5] Initialising ReAct agent ...")
    node_list, _, _ = env.get_pyg_tensors()
    agent = SupplyChainAgent(env, model, guardrail, node_list)

    print("  [5/5] Setting up dashboard ...")
    dashboard = SupplyChainDashboard(max_log_lines=10)

    node_names: Dict[str, str] = {nid: env.nodes[nid].name for nid in env.nodes}

    print(f"\n  Simulation: {args.ticks} ticks  |  "
          f"Failure: {args.fail_node} @ tick {args.fail_tick}  |  "
          f"Severity: {args.severity:.0%}")
    print("\n  Starting in 2 seconds ...\n")
    time.sleep(2)

    # ─── Main Simulation Loop ─────────────────────────────────────────────────
    for tick in range(1, args.ticks + 1):
        try:
            # ── Failure injection ──────────────────────────────────────────────
            if tick == args.fail_tick:
                env.inject_failure(args.fail_node, severity=args.severity)

            # ── Failure resolution (optional) ──────────────────────────────────
            if args.resolve_tick and tick == args.resolve_tick:
                env.resolve_failure(args.fail_node)

            # ── Advance simulation ─────────────────────────────────────────────
            env_state = env.tick()

            # ── Build GNN input tensors ────────────────────────────────────────
            node_list, edge_index, edge_attr = env.get_pyg_tensors()
            graph_data = STGATBlastRadiusModel.build_graph_data(
                env_state, node_list, edge_index, edge_attr
            )

            # ── GNN inference ──────────────────────────────────────────────────
            with torch.no_grad():
                raw_risk_tensor, attn_weights = model(graph_data)

            # ── Blast radius propagation (GNN + heuristic blend) ───────────────
            risk_scores = propagate_blast_radius(
                env_state   = env_state,
                node_list   = node_list,
                gnn_output  = raw_risk_tensor,
                num_hops    = 3,
                blend_alpha = args.blend,
            )

            # ── Write risk scores back into env nodes (for logging) ────────────
            for nid, score in risk_scores.items():
                if nid in env.nodes:
                    env.nodes[nid].risk_score = score

            # ── Agent ReAct cycle ──────────────────────────────────────────────
            cycle = agent.run_cycle(risk_scores, env_state)

            thought_step       = cycle["thought"]
            action_result      = cycle["result"]
            guardrail_blocked  = cycle["guardrail_triggered"]
            guardrail_message  = cycle["guardrail_message"]

            # ── Dashboard refresh ──────────────────────────────────────────────
            dashboard.render(
                tick               = tick,
                risk_scores        = risk_scores,
                node_names         = node_names,
                thought_step       = thought_step,
                action_result      = action_result,
                guardrail_message  = guardrail_message,
                guardrail_blocked  = guardrail_blocked,
                model_params       = param_count,
                failure_node       = (
                    args.fail_node
                    if tick >= args.fail_tick and (
                        not args.resolve_tick or tick < args.resolve_tick
                    )
                    else None
                ),
            )

            # ── Tick delay ────────────────────────────────────────────────────
            if args.delay > 0:
                time.sleep(args.delay)

        except KeyboardInterrupt:
            print("\n\n  [INTERRUPTED] Simulation halted by user.\n")
            break

    # ─── Post-simulation Summary ──────────────────────────────────────────────
    _print_summary(args, agent, guardrail, param_count)


def _print_summary(args, agent, guardrail, param_count: int) -> None:
    mem   = agent.memory
    guard = guardrail.audit_summary()

    W = 80
    print("\n" + "═" * W)
    print("  SIMULATION COMPLETE — POST-RUN SUMMARY")
    print("═" * W)
    print(f"  Failure node injected  : {args.fail_node}  (tick {args.fail_tick}, severity {args.severity:.0%})")
    print(f"  ST-GAT parameters      : {param_count:,}")
    print()
    print("  Agent Performance")
    print("  " + "─" * 40)
    print(f"    Total ReAct cycles   : {agent._tick}")
    print(f"    Actions taken        : {sum(mem.actions_by_type.values())}")
    for atype, cnt in mem.actions_by_type.items():
        print(f"      {atype:<30s}: {cnt}")
    print(f"    Human escalations    : {mem.escalations}")
    print(f"    Cumulative reroute $  : ${mem.cumulative_cost_usd:,.2f}")
    print(f"    Volume rerouted      : {mem.cumulative_vol_rerouted:,.0f} units")
    print()
    print("  Guardrail Statistics")
    print("  " + "─" * 40)
    print(f"    Total intercepted    : {guard['total_intercepted']}")
    print(f"    Blocked (HALT)       : {guard['total_blocked']}")
    print(f"    Auto-approved        : {guard['auto_approved']}")
    print(f"    Human-approved       : {guard['human_approved']}")
    print(f"    Pending queue        : {guard['pending']}")
    if guard["rule_hit_counts"]:
        print(f"    Rule hits:")
        for rule, cnt in guard["rule_hit_counts"].items():
            print(f"      {rule:<35s}: {cnt}")
    print("═" * W + "\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
