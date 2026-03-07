"""
supply_chain/agent.py
======================
Autonomous supply-chain disruption-mitigation agent using the ReAct pattern.

ReAct Loop (per tick)
---------------------
  1. OBSERVE   — ingest GNN risk scores + environment snapshot
  2. REASON    — chain-of-thought analysis of blast radius & root cause
  3. ACT       — call a tool (reroute / reorder / escalate / monitor)
  4. REFLECT   — update memory with outcome and update cost/volume tallies

Tools
-----
  reroute_shipment(order_id, alt_route, cost, volume)
  reorder_inventory(sku, hub_id, quantity)
  request_human_intervention(incident_report)

All tool calls are intercepted by GuardrailLayer before execution.
"""

from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


load_dotenv()


# ─── Enumerations ─────────────────────────────────────────────────────────────

class AgentAction(Enum):
    REROUTE             = "reroute_shipment"
    REORDER             = "reorder_inventory"
    HUMAN_INTERVENTION  = "request_human_intervention"
    MONITOR             = "continue_monitoring"

class RiskLevel(Enum):
    NOMINAL   = "NOMINAL"
    LOW       = "LOW"
    MEDIUM    = "MEDIUM"
    HIGH      = "HIGH"
    CRITICAL  = "CRITICAL"


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ThoughtStep:
    """One complete ReAct cycle record."""
    tick:          int
    observation:   str
    chain_of_thought: List[str]          # step-by-step reasoning trace
    decision:      str                   # concise decision statement
    action:        AgentAction
    action_params: Dict[str, Any]
    timestamp:     float = field(default_factory=time.time)

    @property
    def reasoning(self) -> str:
        """Full reasoning narrative (joined CoT steps)."""
        return "  →  ".join(self.chain_of_thought)


@dataclass
class AgentMemory:
    """Persistent agent state across ticks."""
    risk_snapshots:         List[Dict]        = field(default_factory=list)
    thought_log:            List[ThoughtStep] = field(default_factory=list)
    active_reroutes:        Dict[str, str]    = field(default_factory=dict)
    pending_reorders:       List[str]         = field(default_factory=list)
    cumulative_cost_usd:    float             = 0.0
    cumulative_vol_rerouted:float             = 0.0
    escalations:            int               = 0
    actions_by_type:        Dict[str, int]    = field(default_factory=dict)


# ─── LLM Structured Output Schema ──────────────────────────────────────────────


class AgentDecision(BaseModel):
    """
    Structured decision returned by the LLM "brain".

    This is the single JSON object the model must emit on each tick.
    """

    internal_monologue: List[str] = Field(
        ...,
        description=(
            "Step-by-step reasoning explaining blast radius, options considered, "
            "and why the final action was chosen."
        ),
    )
    chosen_tool: str = Field(
        ...,
        description=(
            "Name of the action to take. One of: "
            "'reroute_shipment', 'reorder_inventory', "
            "'request_human_intervention', 'continue_monitoring'."
        ),
    )
    justification: str = Field(
        ...,
        description="Concise summary of why this action is the best trade-off.",
    )
    target_node_id: Optional[str] = Field(
        default=None,
        description="ID of the primary hub this action targets (e.g. 'HUB-NYC').",
    )


class LLMState(TypedDict, total=False):
    """
    LangGraph state passed through the decision graph.

    Only a single node is used today (LLM planner), but the graph makes this
    extensible for future multi-step workflows.
    """

    tick: int
    observation: str
    risk_scores: Dict[str, float]
    env_snapshot: dict
    decision: AgentDecision


# ─── Tool Implementations ──────────────────────────────────────────────────────

def reroute_shipment(
    order_id:  str,
    alt_route: str,
    cost:      float = 0.0,
    volume:    float = 0.0,
) -> Dict[str, Any]:
    """
    Divert an in-flight shipment to an alternative carrier route.

    Simulates a 90 % success rate; on failure returns a structured error.
    """
    success      = random.random() > 0.10
    eta_delta_h  = random.uniform(1.5, 5.0) if success else 0.0

    return {
        "tool":           "reroute_shipment",
        "order_id":       order_id,
        "alt_route":      alt_route,
        "cost_usd":       cost,
        "volume_units":   volume,
        "success":        success,
        "eta_change_h":   eta_delta_h,
        "message": (
            f"[REROUTE {'OK' if success else 'FAILED'}] "
            f"Order {order_id} → {alt_route}  |  "
            f"ΔCost ${cost:.0f}  |  "
            f"ETA +{eta_delta_h:.1f}h  |  "
            f"Vol {volume:.0f} units"
        ),
    }


def reorder_inventory(
    sku:      str,
    hub_id:   str,
    quantity: int   = 200,
    urgency:  str   = "STANDARD",
) -> Dict[str, Any]:
    """
    Trigger an emergency inventory replenishment for a hub.

    Simulates supplier confirmation latency and cost calculation.
    """
    unit_cost       = random.uniform(0.40, 0.80)
    total_cost      = quantity * unit_cost
    arrival_hours   = random.uniform(18, 52) if urgency == "STANDARD" else random.uniform(8, 20)

    return {
        "tool":           "reorder_inventory",
        "sku":            sku,
        "hub_id":         hub_id,
        "quantity":       quantity,
        "unit_cost_usd":  unit_cost,
        "total_cost_usd": total_cost,
        "urgency":        urgency,
        "eta_hours":      arrival_hours,
        "message": (
            f"[REORDER OK] {quantity}× {sku} → {hub_id}  |  "
            f"${total_cost:.2f}  |  ETA {arrival_hours:.1f}h  |  [{urgency}]"
        ),
    }


def request_human_intervention(incident_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Escalate a decision to a human supply-chain operator.

    Generates an incident record and halts agent execution until approved.
    """
    incident_id = f"INC-{int(time.time()) % 100000:05d}"
    return {
        "tool":        "request_human_intervention",
        "incident_id": incident_id,
        "status":      "PENDING_APPROVAL",
        "report":      incident_report,
        "message": (
            f"[ESCALATION] {incident_id} filed.  "
            f"Reason: {incident_report.get('trigger', 'Unknown')}  |  "
            f"Awaiting operator approval before proceeding."
        ),
        "timestamp": time.time(),
    }


def trigger_manual_override(
    reason: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Explicit manual override tool.

    Used when a human operator declines an LLM-proposed action and chooses to
    intervene outside the autonomous policy (e.g. via phone calls or custom
    playbooks).
    """
    incident_id = f"OVR-{int(time.time()) % 100000:05d}"
    return {
        "tool": "trigger_manual_override",
        "incident_id": incident_id,
        "status": "MANUAL_OVERRIDE",
        "reason": reason,
        "context": context or {},
        "message": (
            f"[MANUAL OVERRIDE] {incident_id} recorded. "
            f"Operator declined autonomous action and will intervene manually."
        ),
        "timestamp": time.time(),
    }


# ─── Risk Classification ──────────────────────────────────────────────────────

_THRESHOLDS = {
    RiskLevel.CRITICAL: 0.82,
    RiskLevel.HIGH:     0.65,
    RiskLevel.MEDIUM:   0.45,
    RiskLevel.LOW:      0.25,
}


def classify_risk(score: float) -> RiskLevel:
    for level, thresh in _THRESHOLDS.items():
        if score >= thresh:
            return level
    return RiskLevel.NOMINAL


# ─── Alternative Route Lookup ─────────────────────────────────────────────────

_ALT_ROUTES: Dict[str, str] = {
    "HUB-NYC": "BOS → PHL → ground-relay bypass",
    "HUB-CHI": "DET → IND → air-charter bypass",
    "HUB-LAX": "SFO → PHX → rail intermodal",
    "HUB-DAL": "HOU → SAT → UPS Freight charter",
    "HUB-ATL": "CHA → BIR → FedEx priority re-sort",
    "HUB-SEA": "PDX → TAC → coast-feeder reroute",
    "HUB-MIA": "ORL → TPA → FedEx express bridge",
    "HUB-DEN": "ABQ → SLC → mountain rail bypass",
    "HUB-PHX": "TUS → LAS → southwest express",
    "HUB-BOS": "MAN → PVD → ground consolidation",
}


def get_alt_route(node_id: str) -> str:
    return _ALT_ROUTES.get(node_id, f"DIRECT-BYPASS-{node_id}")


# ─── ReAct Agent ──────────────────────────────────────────────────────────────

class SupplyChainAgent:
    """
    Stateful ReAct agent for autonomous supply-chain disruption mitigation.

    Each call to `run_cycle(risk_scores, env_state)` executes one full
    Observe → Reason → Act loop and returns a structured result dict.
    """

    def __init__(
        self,
        env,
        model,
        guardrail,
        node_list: List[str],
    ) -> None:
        self.env       = env
        self.model     = model
        self.guardrail = guardrail
        self.node_list = node_list
        self.memory    = AgentMemory()
        self._tick     = 0

        # Lazy-initialised LLM + LangGraph planner
        self._llm = None
        self._llm_structured = None
        self._planner_graph = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run_cycle(
        self,
        risk_scores: Dict[str, float],
        env_state:   dict,
    ) -> Dict[str, Any]:
        """
        Execute one ReAct cycle.

        Returns
        -------
        dict with keys:
          tick, thought (ThoughtStep), result (tool output or None),
          guardrail_triggered (bool), guardrail_message (str)
        """
        self._tick += 1

        # ── 1. Observe ────────────────────────────────────────────────────────
        observation = self._observe(risk_scores, env_state)

        # ── 2. Reason ─────────────────────────────────────────────────────────
        thought = self._reason(observation, risk_scores, env_state)

        # ── 3. Act ────────────────────────────────────────────────────────────
        result, guardrail_triggered, guardrail_msg = self._act(thought)

        # ── 4. Reflect ────────────────────────────────────────────────────────
        self._reflect(thought, result)

        return {
            "tick":                self._tick,
            "thought":             thought,
            "result":              result,
            "guardrail_triggered": guardrail_triggered,
            "guardrail_message":   guardrail_msg,
        }

    # ── ReAct Steps ───────────────────────────────────────────────────────────

    def _observe(self, risk_scores: Dict[str, float], env_state: dict) -> str:
        """Format current world state into an observation string."""
        nodes       = env_state["nodes"]
        failed      = [nid for nid, n in nodes.items() if n.failure_injected]
        risk_sorted = sorted(risk_scores.items(), key=lambda kv: -kv[1])
        top3        = [(nid, nodes[nid].name, s) for nid, s in risk_sorted[:3]
                       if nid in nodes]

        obs_parts = [
            f"Tick {self._tick:03d}",
            f"Active failures: [{', '.join(failed) or 'none'}]",
            f"Peak risk: {risk_sorted[0][1]:.1%} @ {risk_sorted[0][0]}" if risk_sorted else "",
            f"Top-3 at-risk: " + ", ".join(
                f"{name}({score:.0%})" for _, name, score in top3
            ),
            f"Cumulative cost so far: ${self.memory.cumulative_cost_usd:.0f}",
        ]
        return " | ".join(p for p in obs_parts if p)

    def _reason(
        self,
        observation: str,
        risk_scores:  Dict[str, float],
        env_state:    dict,
    ) -> ThoughtStep:
        """
        Reason + decide using a LangGraph-backed Gemini LLM.

        The GNN risk report and environment snapshot are passed into a small
        LangGraph (single-node) which calls Gemini with a structured output
        schema (`AgentDecision`). The JSON it returns is then converted into
        a `ThoughtStep` used by the rest of the agent.

        If the LLM is unavailable or fails, we fall back to the original
        heuristic ReAct policy so the system continues to function.
        """
        # Fast path: no risk information → idle ThoughtStep
        if not risk_scores:
            return self._idle_step(observation)

        # Try the LLM planner; on any error, fall back to heuristic logic.
        try:
            decision = self._llm_decide(observation, risk_scores, env_state)
            return self._decision_to_thought(decision, observation, risk_scores, env_state)
        except Exception:
            # Fallback: deterministic heuristic behaviour from the original agent
            return self._heuristic_reason(observation, risk_scores, env_state)

    # ── LLM-backed planner helpers ─────────────────────────────────────────────

    def _ensure_llm_planner(self) -> None:
        """Initialise the Gemini LLM and LangGraph planner graph if needed."""
        if self._planner_graph is not None:
            return

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. "
                "Set it in your environment or a .env file to enable the LLM agent."
            )

        # Low-temperature, deterministic-ish logistics planner
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.2,
            max_output_tokens=1024,
        )
        # Enforce JSON-shaped responses matching AgentDecision
        self._llm_structured = self._llm.with_structured_output(AgentDecision)

        graph = StateGraph(LLMState)

        def decide_node(state: LLMState) -> LLMState:
            """Single-node planner that calls Gemini with a structured schema."""
            observation = state["observation"]
            risk_scores = state["risk_scores"]
            env_snapshot = state["env_snapshot"]
            tick = state["tick"]

            # Build a compact, model-friendly risk report
            nodes = env_snapshot["nodes"]
            failed = [nid for nid, n in nodes.items() if n.failure_injected]
            risk_sorted = sorted(risk_scores.items(), key=lambda kv: -kv[1])
            top_rows = []
            for nid, score in risk_sorted[:5]:
                node = nodes.get(nid)
                name = node.name if node else nid
                top_rows.append(
                    f"- {nid} ({name}): risk={score:.1%}, "
                    f"backlog={getattr(node, 'current_backlog', 0):.0f}, "
                    f"health={getattr(node, 'health_status', 1.0):.0%}, "
                    f"volume={getattr(node, 'incoming_volume', 0):.0f}"
                )

            risk_report = "\n".join(top_rows) or "No nodes available."
            failed_str = ", ".join(failed) if failed else "none"

            prompt = (
                "You are an autonomous logistics incident-response agent operating a "
                "global parcel network. A Graph Neural Network (GNN) provides a per-hub "
                "blast-radius risk score in [0, 1]. Your job is to interpret the GNN "
                "signal and choose ONE operational tool to call this tick.\n\n"
                "Tools you can conceptually choose between:\n"
                "  - reroute_shipment  → emergency diversion of in-flight volume\n"
                "  - reorder_inventory → pre-position inventory to absorb disruption\n"
                "  - request_human_intervention → escalate ambiguous / high-blast events\n"
                "  - continue_monitoring → take no action this tick\n\n"
                "Return **only** a JSON object matching this schema, with no extra text:\n"
                "{\n"
                '  "internal_monologue": ["string", "..."],\n'
                '  "chosen_tool": "reroute_shipment | reorder_inventory | '
                'request_human_intervention | continue_monitoring",\n'
                '  "justification": "string",\n'
                '  "target_node_id": "HUB-XXX or null"\n'
                "}\n\n"
                f"Current tick: {tick}\n"
                f"Observation: {observation}\n"
                f"Failed hubs: {failed_str}\n"
                "Top risk report (highest first):\n"
                f"{risk_report}\n\n"
                "Optimise for protecting downstream customer promise while keeping cost "
                "and blast radius under control. Prefer 'continue_monitoring' when all "
                "risks are nominal. Prefer 'request_human_intervention' when the blast "
                "radius or cost is very high and you are uncertain."
            )

            decision = self._llm_structured.invoke(prompt)
            return {
                **state,
                "decision": decision,
            }

        graph.add_node("decide", decide_node)
        graph.set_entry_point("decide")
        graph.add_edge("decide", END)
        self._planner_graph = graph.compile()

    def _llm_decide(
        self,
        observation: str,
        risk_scores: Dict[str, float],
        env_state: dict,
    ) -> AgentDecision:
        """Run one LangGraph LLM planning cycle and return the structured decision."""
        self._ensure_llm_planner()

        out: LLMState = self._planner_graph.invoke(
            {
                "tick": self._tick,
                "observation": observation,
                "risk_scores": risk_scores,
                "env_snapshot": env_state,
            }
        )
        decision = out.get("decision")
        if decision is None:
            raise RuntimeError("LLM planner returned no decision.")
        return decision

    def _decision_to_thought(
        self,
        decision: AgentDecision,
        observation: str,
        risk_scores: Dict[str, float],
        env_state: dict,
    ) -> ThoughtStep:
        """
        Convert an `AgentDecision` JSON object into the agent's `ThoughtStep`.

        The LLM owns the chain-of-thought (internal_monologue), chosen tool,
        and justification; this method maps that onto our existing action
        schema and fills in any missing operational parameters.
        """
        tool = decision.chosen_tool.strip()
        tool = tool.lower().replace(" ", "_")

        # Map string tool name → AgentAction
        if tool == "reroute_shipment":
            action = AgentAction.REROUTE
        elif tool == "reorder_inventory":
            action = AgentAction.REORDER
        elif tool == "request_human_intervention":
            action = AgentAction.HUMAN_INTERVENTION
        else:
            # Default safe behaviour is monitoring-only
            action = AgentAction.MONITOR

        nodes = env_state["nodes"]
        target_id = decision.target_node_id or (
            max(risk_scores, key=risk_scores.get) if risk_scores else None
        )
        target_name = (
            nodes[target_id].name if target_id and target_id in nodes else target_id
        )

        # Fill in sensible defaults for tool parameters so the rest of the system
        # (guardrails, tools, dashboard) continues to operate unchanged.
        if action == AgentAction.REROUTE and target_id:
            high_risk_nodes = [
                nid for nid, s in risk_scores.items() if s >= 0.65 and nid in nodes
            ]
            est_cost = max(250.0, 185.0 * max(1, len(high_risk_nodes)))
            est_volume = sum(
                nodes[n].incoming_volume for n in high_risk_nodes
            ) or nodes[target_id].incoming_volume

            action_params = {
                "order_id": f"ORD-{self._tick:04d}-{target_id[-3:] if len(target_id) >= 3 else target_id}",
                "alt_route": get_alt_route(target_id),
                "affected_nodes": high_risk_nodes or [target_id],
                "estimated_cost": float(est_cost),
                "volume": float(est_volume),
                "priority": "URGENT",
            }
            decision_text = decision.justification

        elif action == AgentAction.REORDER and target_id:
            sku = f"SKU-{abs(hash(target_id)) % 9000 + 1000}"
            qty = 250
            action_params = {
                "sku": sku,
                "hub_id": target_id,
                "quantity": qty,
                "urgency": "STANDARD",
            }
            decision_text = decision.justification

        elif action == AgentAction.HUMAN_INTERVENTION:
            action_params = {
                "reason": decision.justification,
                "target_node": target_id,
            }
            decision_text = decision.justification or "Escalating to human operator."

        else:
            # Monitoring-only
            action_params = {}
            decision_text = decision.justification or "System nominal. No action taken."

        thought = ThoughtStep(
            tick=self._tick,
            observation=observation,
            chain_of_thought=list(decision.internal_monologue),
            decision=decision_text,
            action=action,
            action_params=action_params,
        )
        self.memory.thought_log.append(thought)
        return thought

    # ── Original heuristic policy (fallback) ────────────────────────────────────

    def _heuristic_reason(
        self,
        observation: str,
        risk_scores: Dict[str, float],
        env_state: dict,
    ) -> ThoughtStep:
        """
        Multi-step chain-of-thought reasoning.

        Produces a `ThoughtStep` with a complete reasoning trace and
        a selected action + parameters.
        """
        nodes    = env_state["nodes"]
        edges    = env_state["edges"]
        failed   = [nid for nid, n in nodes.items() if n.failure_injected]

        peak_node  = max(risk_scores, key=risk_scores.get)
        peak_score = risk_scores[peak_node]
        peak_level = classify_risk(peak_score)
        peak_name  = nodes[peak_node].name if peak_node in nodes else peak_node

        high_risk  = {n: s for n, s in risk_scores.items()
                      if classify_risk(s) in (RiskLevel.HIGH, RiskLevel.CRITICAL)}
        medium_risk= {n: s for n, s in risk_scores.items()
                      if classify_risk(s) == RiskLevel.MEDIUM}

        # ── Chain of Thought ──────────────────────────────────────────────────
        cot: List[str] = []

        cot.append(
            f"GNN blast-radius scan at tick {self._tick} complete.  "
            f"Peak risk score: {peak_score:.1%} ({peak_level.value}) at {peak_name}."
        )

        if failed:
            failed_names = [nodes[f].name for f in failed if f in nodes]
            cot.append(
                f"Root-cause tracing: failure(s) detected at [{', '.join(failed_names)}].  "
                f"Signal propagated across {len(high_risk)} downstream hub(s) within 3-hop blast radius."
            )

        if peak_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            # Estimate time-to-impact (lower risk → more time headroom)
            tti_h = max(1, round((1.0 - peak_score) * 8))
            at_risk_vol = sum(
                nodes[n].incoming_volume for n in high_risk if n in nodes
            )
            cot.append(
                f"Time-to-impact estimate: ~{tti_h}h based on risk gradient.  "
                f"Total volume at risk: ~{at_risk_vol:.0f} units across blast radius."
            )

            # Check if any high-risk node has cascading edge delays
            cascade_edges = [
                e for e in edges
                if e.source in high_risk and e.current_delay > 3.0
            ]
            if cascade_edges:
                worst = max(cascade_edges, key=lambda e: e.current_delay)
                cot.append(
                    f"Critical route [{worst.source} → {worst.target}] shows "
                    f"{worst.current_delay:.1f}h delay via {worst.carrier}.  "
                    f"Route reliability degraded to {worst.route_reliability:.0%}."
                )

            est_cost  = len(high_risk) * 185.0
            at_risk_v = sum(nodes[n].incoming_volume for n in high_risk if n in nodes)

            cot.append(
                f"Decision: REROUTE protocol initiated for {peak_name}.  "
                f"Estimated incremental cost: ${est_cost:.0f}.  "
                f"Alt route: {get_alt_route(peak_node)}."
            )
            decision = (
                f"GNN predicts {peak_score:.0%} risk at {peak_name} in ~{tti_h}h. "
                f"Initiating reroute for all shipments transiting blast radius."
            )
            action       = AgentAction.REROUTE
            action_params= {
                "order_id":       f"ORD-{self._tick:04d}-{peak_node[-3:]}",
                "alt_route":      get_alt_route(peak_node),
                "affected_nodes": list(high_risk.keys()),
                "estimated_cost": est_cost,
                "volume":         at_risk_v,
                "priority":       "URGENT",
            }

        elif peak_level == RiskLevel.MEDIUM or medium_risk:
            target_node = max(medium_risk, key=medium_risk.get) if medium_risk else peak_node
            target_name = nodes[target_node].name if target_node in nodes else target_node
            sku         = f"SKU-{abs(hash(target_node)) % 9000 + 1000}"
            qty         = 250

            cot.append(
                f"No imminent critical failure.  "
                f"{target_name} shows MEDIUM risk ({risk_scores.get(target_node, 0):.0%}).  "
                f"Pre-positioning inventory buffer to absorb potential disruption."
            )
            cot.append(
                f"Decision: REORDER {qty} units of {sku} to {target_name}.  "
                f"Sufficient inventory headroom prevents cascade if risk elevates."
            )
            decision     = f"Proactive reorder for {target_name} to hedge {peak_level.value} risk."
            action       = AgentAction.REORDER
            action_params= {
                "sku":      sku,
                "hub_id":   target_node,
                "quantity": qty,
                "urgency":  "STANDARD",
            }

        else:
            cot.append(
                f"All nodes within nominal risk parameters.  "
                f"No intervention required.  Passive monitoring active."
            )
            decision     = "System nominal. No action taken."
            action       = AgentAction.MONITOR
            action_params= {}

        thought = ThoughtStep(
            tick             = self._tick,
            observation      = observation,
            chain_of_thought = cot,
            decision         = decision,
            action           = action,
            action_params    = action_params,
        )
        self.memory.thought_log.append(thought)
        return thought

    def _act(
        self, thought: ThoughtStep
    ) -> Tuple[Optional[Dict], bool, str]:
        """
        Execute the chosen action after passing through the guardrail layer.

        Returns (tool_result, guardrail_triggered, guardrail_message).
        """
        if thought.action == AgentAction.MONITOR:
            return None, False, "PASS — no action required"

        command = {
            "action": thought.action.value,
            "params": thought.action_params,
            "tick":   self._tick,
        }

        approved, guard_result = self.guardrail.intercept(command)

        if not approved:
            # Guardrail blocked — escalate to human
            incident = {
                "trigger":        "Guardrail threshold exceeded",
                "action":         thought.action.value,
                "params":         thought.action_params,
                "reasoning":      thought.reasoning,
                "guardrail_rules": guard_result.triggered_rules,
                "tick":           self._tick,
            }
            result = request_human_intervention(incident)
            self.memory.escalations += 1
            return result, True, " | ".join(guard_result.triggered_rules)

        # Execute approved action
        if thought.action == AgentAction.REROUTE:
            result = reroute_shipment(
                order_id  = thought.action_params["order_id"],
                alt_route = thought.action_params["alt_route"],
                cost      = thought.action_params.get("estimated_cost", 0.0),
                volume    = thought.action_params.get("volume", 0.0),
            )
            self.memory.cumulative_cost_usd     += thought.action_params.get("estimated_cost", 0.0)
            self.memory.cumulative_vol_rerouted += thought.action_params.get("volume", 0.0)

        elif thought.action == AgentAction.REORDER:
            result = reorder_inventory(
                sku      = thought.action_params["sku"],
                hub_id   = thought.action_params["hub_id"],
                quantity = thought.action_params.get("quantity", 200),
                urgency  = thought.action_params.get("urgency", "STANDARD"),
            )

        else:
            result = None

        self.memory.actions_by_type[thought.action.value] = (
            self.memory.actions_by_type.get(thought.action.value, 0) + 1
        )
        return result, False, "PASS — within policy limits"

    def _reflect(self, thought: ThoughtStep, result: Optional[Dict]) -> None:
        """Update memory with cycle outcome (for future reasoning context)."""
        if result:
            # Log last action outcome so future reasoning can reference it
            self.memory.risk_snapshots = self.memory.risk_snapshots[-20:]  # keep last 20

    def _idle_step(self, observation: str) -> ThoughtStep:
        return ThoughtStep(
            tick              = self._tick,
            observation       = observation,
            chain_of_thought  = ["No risk data available. Standing by."],
            decision          = "Idle — awaiting GNN output.",
            action            = AgentAction.MONITOR,
            action_params     = {},
        )
