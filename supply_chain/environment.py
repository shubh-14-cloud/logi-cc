"""
supply_chain/environment.py
============================
Discrete-event logistics simulation.

World model:
  Nodes  — Warehouses / Distribution Hubs
  Edges  — Carrier routes between hubs

Every `tick()` call advances simulation time by 1 hour and updates:
  • Node telemetry : throughput, backlog, health score
  • Edge telemetry : route reliability, current delay, traffic noise

Failure injection injects a severity-weighted shock at a chosen node
and propagates degradation to all outbound edges.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import simpy


# ─── Data Containers ─────────────────────────────────────────────────────────

@dataclass
class NodeState:
    """Live telemetry for a single supply-chain hub."""

    node_id:         str
    name:            str
    region:          str

    # GNN input features
    current_backlog:  float   # absolute units queued          [0 – 1000]
    health_status:    float   # operational health             [0.0 – 1.0]
    incoming_volume:  float   # inbound units / hour
    priority_index:   float   # business criticality weight    [0.0 – 1.0]

    # Derived / operational
    throughput:       float   # units processed / hour
    failure_injected: bool    = False
    failure_severity: float   = 0.0
    risk_score:       float   = 0.0   # written by GNN each tick

    def to_feature_vec(self) -> List[float]:
        """Return normalised 4-dim feature vector for GNN input."""
        return [
            min(self.current_backlog / 1000.0, 1.0),
            self.health_status,
            min(self.incoming_volume / 600.0, 1.0),
            self.priority_index,
        ]


@dataclass
class EdgeState:
    """Live telemetry for a carrier route between two hubs."""

    source:            str
    target:            str
    carrier:           str

    # GNN edge features
    route_reliability: float   # on-time delivery rate     [0.0 – 1.0]
    current_delay:     float   # extra delay in hours      [0 – ∞)
    distance_weight:   float   # normalised great-circle   [0.0 – 1.0]

    # Operational metadata
    typical_lead_time: float   # baseline hours
    daily_capacity:    float   # max units / day

    def to_feature_vec(self) -> List[float]:
        """Return normalised 3-dim feature vector for GNN edge input."""
        return [
            self.route_reliability,
            min(self.current_delay / 24.0, 1.0),
            self.distance_weight,
        ]


# ─── Environment ─────────────────────────────────────────────────────────────

class LogisticsEnvironment:
    """
    Class-based discrete-event supply-chain simulator backed by SimPy.

    Usage
    -----
    env = LogisticsEnvironment(seed=42)
    env.inject_failure("HUB-NYC", severity=0.8)
    state = env.tick()          # advance 1 simulated hour
    node_list, ei, ea = env.get_pyg_tensors()
    """

    # Hub definitions: (id, name, region, init_health, init_volume, priority)
    _HUB_DEFS: List[Tuple] = [
        ("HUB-NYC", "New York",     "NE",  0.90, 480, 0.92),
        ("HUB-CHI", "Chicago",      "MW",  0.86, 340, 0.78),
        ("HUB-LAX", "Los Angeles",  "W",   0.88, 390, 0.83),
        ("HUB-DAL", "Dallas",       "S",   0.82, 295, 0.71),
        ("HUB-ATL", "Atlanta",      "SE",  0.87, 360, 0.74),
        ("HUB-SEA", "Seattle",      "NW",  0.80, 230, 0.66),
        ("HUB-MIA", "Miami",        "SE",  0.84, 255, 0.69),
        ("HUB-DEN", "Denver",       "MW",  0.79, 190, 0.61),
        ("HUB-PHX", "Phoenix",      "SW",  0.81, 210, 0.63),
        ("HUB-BOS", "Boston",       "NE",  0.85, 275, 0.72),
    ]

    # Route definitions: (src, tgt, carrier, reliability, lead_time_h, dist_norm, capacity/day)
    _ROUTE_DEFS: List[Tuple] = [
        ("HUB-NYC", "HUB-CHI", "UPS-Air",   0.93, 12, 0.45, 2000),
        ("HUB-NYC", "HUB-ATL", "FedEx",     0.88, 18, 0.35, 1800),
        ("HUB-NYC", "HUB-BOS", "UPS-Grnd",  0.95,  4, 0.10, 1200),
        ("HUB-NYC", "HUB-MIA", "DHL",       0.85, 24, 0.52, 1500),
        ("HUB-CHI", "HUB-DAL", "FedEx",     0.85, 20, 0.50, 1600),
        ("HUB-CHI", "HUB-DEN", "UPS-Air",   0.80, 22, 0.55, 1400),
        ("HUB-CHI", "HUB-LAX", "XPO",       0.79, 36, 0.70, 1000),
        ("HUB-LAX", "HUB-SEA", "UPS-Air",   0.91, 14, 0.30, 1700),
        ("HUB-LAX", "HUB-DEN", "FedEx",     0.83, 16, 0.40, 1300),
        ("HUB-LAX", "HUB-PHX", "UPS-Grnd",  0.90,  8, 0.18, 1100),
        ("HUB-DAL", "HUB-ATL", "FedEx",     0.87, 12, 0.38, 1500),
        ("HUB-DAL", "HUB-MIA", "DHL",       0.82, 16, 0.42, 1200),
        ("HUB-ATL", "HUB-MIA", "UPS-Air",   0.92,  8, 0.28, 1600),
        ("HUB-SEA", "HUB-DEN", "XPO",       0.77, 18, 0.48, 1100),
        ("HUB-DEN", "HUB-PHX", "UPS-Grnd",  0.83, 10, 0.22, 1000),
        ("HUB-BOS", "HUB-NYC", "UPS-Grnd",  0.94,  4, 0.10, 1200),
    ]

    def __init__(self, seed: int = 42):
        self.rng   = np.random.RandomState(seed)
        self._rng  = random.Random(seed)
        self.sim   = simpy.Environment()     # SimPy clock (1 tick = 1 simulated hour)
        self.tick_count: int = 0

        self.nodes: Dict[str, NodeState] = {}
        self.edges: List[EdgeState]      = []

        self._build_network()

    # ── Construction ─────────────────────────────────────────────────────────

    def _build_network(self) -> None:
        for (nid, name, region, health, volume, priority) in self._HUB_DEFS:
            backlog = float(self.rng.uniform(40, 180))
            self.nodes[nid] = NodeState(
                node_id        = nid,
                name           = name,
                region         = region,
                current_backlog= backlog,
                health_status  = health,
                incoming_volume= float(volume) + self.rng.normal(0, 15),
                priority_index = priority,
                throughput     = volume * 0.96,
            )

        for (src, tgt, carrier, rel, lt, dist, cap) in self._ROUTE_DEFS:
            self.edges.append(EdgeState(
                source            = src,
                target            = tgt,
                carrier           = carrier,
                route_reliability = rel,
                current_delay     = float(self.rng.uniform(0, 1.5)),
                distance_weight   = dist,
                typical_lead_time = lt,
                daily_capacity    = cap,
            ))

    # ── Failure Injection ─────────────────────────────────────────────────────

    def inject_failure(self, node_id: str, severity: float = 0.80) -> None:
        """
        Inject a disruption at `node_id`.

        Effects:
          • Node health degrades by severity × current_health
          • Node backlog spikes (inbound continues, throughput drops)
          • All outbound edges lose reliability and accrue delay
        """
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node '{node_id}'. "
                             f"Valid IDs: {list(self.nodes.keys())}")

        node = self.nodes[node_id]
        node.failure_injected = True
        node.failure_severity = severity
        node.health_status    = max(0.02, node.health_status * (1.0 - severity))
        node.current_backlog += node.incoming_volume * severity * 3.0   # hours of jammed inflow

        for edge in self.edges:
            if edge.source == node_id:
                edge.route_reliability *= max(0.10, 1.0 - severity * 0.70)
                edge.current_delay     += float(self.rng.uniform(4.0, 8.0) * severity)

    # ── Simulation Step ───────────────────────────────────────────────────────

    def tick(self) -> dict:
        """
        Advance simulation by 1 hour.

        Returns the full state snapshot after update.
        """
        self.tick_count += 1
        self.sim.step() if self.sim.peek() < float("inf") else None

        self._update_nodes()
        self._update_edges()

        return self.get_state_snapshot()

    def _update_nodes(self) -> None:
        for node in self.nodes.values():
            noise = float(self.rng.normal(0, 0.015))

            if node.failure_injected:
                # Progressive health decay while failure is active
                decay = 0.04 * node.failure_severity
                node.health_status = max(0.02, node.health_status - decay + noise * 0.5)
            else:
                # Natural recovery
                node.health_status = min(1.0, node.health_status + 0.008 + noise)

            # Backlog dynamics: inflow – (throughput × health)
            effective_throughput = node.throughput * node.health_status
            volume_noise         = float(self.rng.normal(0, 18))
            node.current_backlog = max(
                0.0,
                node.current_backlog
                + node.incoming_volume
                - effective_throughput
                + volume_noise,
            )

            # Demand fluctuation (slow-moving)
            node.incoming_volume = max(
                20.0,
                node.incoming_volume + float(self.rng.normal(0, 12)),
            )

    def _update_edges(self) -> None:
        for edge in self.edges:
            # Delay mean-reverts toward 0
            mean_reversion = -edge.current_delay * 0.15
            shock          = float(self.rng.normal(mean_reversion, 0.4))
            edge.current_delay = max(0.0, edge.current_delay + shock)

            # Reliability slowly recovers unless source node is still failed
            src = self.nodes.get(edge.source)
            if src and src.failure_injected:
                edge.route_reliability = max(
                    0.05,
                    edge.route_reliability - 0.01 * src.failure_severity,
                )
            else:
                edge.route_reliability = min(
                    0.99,
                    edge.route_reliability + float(self.rng.uniform(0.003, 0.008)),
                )

    # ── Tensor Export for GNN ─────────────────────────────────────────────────

    def get_pyg_tensors(self) -> Tuple[List[str], List[List[int]], List[List[float]]]:
        """
        Return (node_list, edge_index, edge_attr) in PyG-compatible format.

        edge_index : [[src_i, tgt_i], ...]  — will be transposed in model
        edge_attr  : [[reliability, delay_norm, dist_weight], ...]
        """
        node_list = list(self.nodes.keys())
        idx       = {nid: i for i, nid in enumerate(node_list)}

        edge_index: List[List[int]]   = []
        edge_attr:  List[List[float]] = []

        for e in self.edges:
            if e.source in idx and e.target in idx:
                edge_index.append([idx[e.source], idx[e.target]])
                edge_attr.append(e.to_feature_vec())

        return node_list, edge_index, edge_attr

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> dict:
        return {
            "tick":  self.tick_count,
            "nodes": {nid: node for nid, node in self.nodes.items()},
            "edges": self.edges,
        }

    def resolve_failure(self, node_id: str) -> None:
        """Mark a failure as resolved (recovery continues naturally)."""
        if node_id in self.nodes:
            self.nodes[node_id].failure_injected = False
            self.nodes[node_id].failure_severity  = 0.0
