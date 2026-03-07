"""
Agentic Supply Chain Layer
==========================
ST-GAT Blast Radius Prediction + Autonomous Disruption Mitigation Agent

Components:
  environment  — Discrete-event logistics simulation (SimPy + class state machine)
  gnn_model    — Spatio-Temporal Graph Attention Network (3-layer GAT, ~150k params)
  agent        — ReAct-pattern autonomous agent with tool-calling
  guardrails   — Human-in-the-loop safety interception layer
  dashboard    — Rich CLI live dashboard
"""

from .environment import LogisticsEnvironment, NodeState, EdgeState
from .gnn_model   import STGATBlastRadiusModel
from .agent       import SupplyChainAgent, reroute_shipment, reorder_inventory, request_human_intervention
from .guardrails  import GuardrailLayer
from .dashboard   import SupplyChainDashboard

__all__ = [
    "LogisticsEnvironment", "NodeState", "EdgeState",
    "STGATBlastRadiusModel",
    "SupplyChainAgent", "reroute_shipment", "reorder_inventory", "request_human_intervention",
    "GuardrailLayer",
    "SupplyChainDashboard",
]
