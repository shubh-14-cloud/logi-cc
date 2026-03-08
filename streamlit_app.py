from __future__ import annotations

import random
import time
from typing import Dict, Optional

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go

from supply_chain.environment import LogisticsEnvironment
from supply_chain.gnn_model import (
    STGATBlastRadiusModel,
    propagate_blast_radius,
    initialise_gnn_weights,
)
from supply_chain.agent import SupplyChainAgent, AgentAction
from supply_chain.guardrails import GuardrailLayer, COST_APPROVAL_USD
from supply_chain.hub_coords import HUB_GEO, HUB_ROUTES, risk_to_hex, risk_label


st.set_page_config(
    page_title="Agentic Supply Chain Command Center",
    layout="wide",
)


# ─── Session-scoped initialisation ─────────────────────────────────────────────


def init_simulation(seed: int = 42, use_trained_gnn: bool = True) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = LogisticsEnvironment(seed=seed)

    model = STGATBlastRadiusModel(
        node_feature_dim=4,
        edge_feature_dim=3,
        hidden_dim=64,
        num_heads=4,
        dropout=0.10,
    )
    model.eval()
    # Load trained weights when available (or fall back to heuristic warm-start)
    initialise_gnn_weights(model, prefer_trained=use_trained_gnn)

    total_volume = sum(n.incoming_volume for n in env.nodes.values())
    guardrail = GuardrailLayer(
        cost_threshold_usd=COST_APPROVAL_USD,
        volume_threshold_pct=0.10,
        total_network_volume=total_volume,
        auto_approve=False,  # real human-in-the-loop from Streamlit UI
    )

    node_list, _, _ = env.get_pyg_tensors()
    agent = SupplyChainAgent(env, model, guardrail, node_list)

    st.session_state["env"] = env
    st.session_state["model"] = model
    st.session_state["guardrail"] = guardrail
    st.session_state["agent"] = agent
    st.session_state["node_list"] = node_list
    st.session_state["param_count"] = model.count_parameters()
    st.session_state["last_cycle"] = None
    st.session_state["pending_action"] = None
    st.session_state["tick"] = 0
    st.session_state["use_trained_gnn"] = use_trained_gnn


if "env" not in st.session_state:
    # Default to using trained weights if available
    init_simulation(use_trained_gnn=True)


# ─── Sidebar Controls ──────────────────────────────────────────────────────────


with st.sidebar:
    st.markdown("### Control Panel")
    fail_node = st.selectbox(
        "Failure hub",
        options=list(HUB_GEO.keys()),
        index=0,
    )
    fail_tick = st.number_input("Inject failure at tick", min_value=1, max_value=100, value=3)
    severity = st.slider("Failure severity", min_value=0.1, max_value=1.0, value=0.8, step=0.05)
    resolve_tick: Optional[int] = st.number_input(
        "Resolve failure at tick (0 = never)", min_value=0, max_value=100, value=0
    ) or None

    st.markdown("---")
    gnn_mode = st.radio(
        "GNN Weight Mode",
        options=["Use Trained GNN Weights", "Use Heuristic Weights"],
        index=0,
    )
    use_trained_gnn = gnn_mode == "Use Trained GNN Weights"

    st.markdown("---")
    seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Reset Simulation", use_container_width=True):
            init_simulation(seed=seed, use_trained_gnn=use_trained_gnn)
            st.rerun()
    with col_btn2:
        next_tick_clicked = st.button("Next Tick ▶", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Human-in-the-loop policy**  \n"
        f"- Any action with estimated cost **> ${COST_APPROVAL_USD:.0f}** "
        "will pause execution and require operator approval."
    )


env: LogisticsEnvironment = st.session_state["env"]
model: STGATBlastRadiusModel = st.session_state["model"]
guardrail: GuardrailLayer = st.session_state["guardrail"]
agent: SupplyChainAgent = st.session_state["agent"]
param_count: int = st.session_state["param_count"]


# ─── Simulation Step (without side effects until approved) ─────────────────────


def run_agent_tick() -> None:
    """Advance the world by one tick and let the LLM agent decide an action.

    Potential high-cost actions are held for explicit human approval.
    """
    if st.session_state.get("pending_action"):
        # Do not advance simulation if we are still waiting for approval
        return

    st.session_state["tick"] += 1
    tick = st.session_state["tick"]

    # Inject / resolve failures based on configured ticks
    if tick == fail_tick:
        env.inject_failure(fail_node, severity=severity)
    if resolve_tick and tick == resolve_tick:
        env.resolve_failure(fail_node)

    # Advance environment + build GNN inputs
    env_state = env.tick()
    node_list, edge_index, edge_attr = env.get_pyg_tensors()
    graph_data = STGATBlastRadiusModel.build_graph_data(env_state, node_list, edge_index, edge_attr)

    with torch.no_grad():
        raw_risk_tensor, _ = model(graph_data)

    risk_scores = propagate_blast_radius(
        env_state=env_state,
        node_list=node_list,
        gnn_output=raw_risk_tensor,
        num_hops=3,
        blend_alpha=0.45,
    )

    # Write risk scores back into environment nodes for downstream visuals
    for nid, score in risk_scores.items():
        if nid in env.nodes:
            env.nodes[nid].risk_score = score

    # Run Observe + Reason via the LangGraph LLM planner
    agent._tick += 1  # Internal tick counter (kept in sync with env)
    observation = agent._observe(risk_scores, env_state)
    thought = agent._reason(observation, risk_scores, env_state)

    # Check whether we need to pause for human approval
    est_cost = float(thought.action_params.get("estimated_cost", 0.0))
    requires_cost_approval = (
        thought.action not in (AgentAction.MONITOR,)
        and est_cost > COST_APPROVAL_USD
    )
    requires_human_tool = thought.action == AgentAction.HUMAN_INTERVENTION

    if requires_cost_approval or requires_human_tool:
        st.session_state["pending_action"] = {
            "tick": tick,
            "thought": thought,
            "env_state": env_state,
            "risk_scores": risk_scores,
            "estimated_cost": est_cost,
        }
        if requires_human_tool:
            guardrail_msg = "Agent requested human intervention; awaiting operator decision."
        else:
            guardrail_msg = (
                f"Awaiting human approval for high-cost action "
                f"(${est_cost:.0f} > ${COST_APPROVAL_USD:.0f})."
            )

        st.session_state["last_cycle"] = {
            "tick": tick,
            "risk_scores": risk_scores,
            "thought": thought,
            "result": None,
            "guardrail_triggered": True,
            "guardrail_message": guardrail_msg,
        }
    else:
        # Safe to execute immediately via the normal guardrail layer
        result, guardrail_blocked, guardrail_msg = agent._act(thought)
        st.session_state["last_cycle"] = {
            "tick": tick,
            "risk_scores": risk_scores,
            "thought": thought,
            "result": result,
            "guardrail_triggered": guardrail_blocked,
            "guardrail_message": guardrail_msg,
        }


if next_tick_clicked:
    run_agent_tick()


last_cycle = st.session_state.get("last_cycle")
pending = st.session_state.get("pending_action")


# ─── Human Approval Modal / Expander ───────────────────────────────────────────


if pending:
    thought = pending["thought"]
    est_cost = pending["estimated_cost"]
    tick = pending["tick"]
    is_human_tool = thought.action == AgentAction.HUMAN_INTERVENTION

    title = (
        "👤 Human Intervention Requested — Approve Escalation"
        if is_human_tool
        else "⚠ Human Approval Required — High-Cost Action"
    )

    with st.expander(title, expanded=True):
        thought = pending["thought"]
        est_cost = pending["estimated_cost"]
        tick = pending["tick"]

        if is_human_tool:
            st.markdown(
                f"**Agent has requested human intervention on tick {tick}.**  \n"
                f"Proposed escalation action: `{thought.action.value}`"
            )
        else:
            st.markdown(
                f"**Proposed action (tick {tick}):** `{thought.action.value}`  \n"
                f"**Estimated incremental cost:** `${est_cost:,.0f}`  \n"
                f"**Policy threshold:** `${COST_APPROVAL_USD:,.0f}`"
            )

        st.markdown("**LLM Justification**")
        st.info(thought.decision)

        st.markdown("**LLM Internal Monologue (Chain of Thought)**")
        for i, step in enumerate(thought.chain_of_thought, 1):
            st.write(f"{i}. {step}")

        col_a, col_b = st.columns(2)
        with col_a:
            approve = st.button("✅ Approve and Execute", use_container_width=True)
        with col_b:
            decline = st.button("❌ Decline / Manual Override", use_container_width=True)

        if approve:
            result, guardrail_blocked, guardrail_msg = agent._act(thought)
            st.session_state["last_cycle"] = {
                "tick": tick,
                "risk_scores": pending["risk_scores"],
                "thought": thought,
                "result": result,
                "guardrail_triggered": guardrail_blocked,
                "guardrail_message": guardrail_msg,
            }
            st.session_state["pending_action"] = None
            st.rerun()

        if decline:
            # Treat this as an explicit manual override — no reroute executed.
            from supply_chain.agent import trigger_manual_override

            override_reason = (
                "Operator declined LLM escalation request; handling manually."
                if is_human_tool
                else "Operator declined high-cost autonomous action."
            )
            override_context = {
                "tick": tick,
                "proposed_action": thought.action.value,
                "proposed_params": thought.action_params,
            }
            if not is_human_tool:
                override_context["estimated_cost"] = est_cost

            override_result = trigger_manual_override(
                reason=override_reason,
                context=override_context,
            )

            st.session_state["last_cycle"] = {
                "tick": tick,
                "risk_scores": pending["risk_scores"],
                "thought": thought,
                "result": override_result,
                "guardrail_triggered": True,
                "guardrail_message": (
                    "Operator declined LLM proposal; manual override recorded."
                ),
            }
            st.session_state["pending_action"] = None
            st.rerun()


# ─── Main Layout ───────────────────────────────────────────────────────────────


st.markdown("## Agentic Supply Chain Command Center")
st.caption(
    "GNN blast-radius prediction • Gemini LLM decisioning • Human-in-the-loop guardrails"
)

col_left, col_right = st.columns([2, 1])


def build_map_figure(risk_scores: Dict[str, float]) -> go.Figure:
    fig = go.Figure()

    # Hub markers
    lats = []
    lons = []
    texts = []
    colors = []
    sizes = []

    for hub_id, meta in HUB_GEO.items():
        score = risk_scores.get(hub_id, 0.0)
        lats.append(meta["lat"])
        lons.append(meta["lon"])
        colors.append(risk_to_hex(score))
        sizes.append(10 + 10 * score)
        texts.append(
            f"{meta['city']}, {meta['state']}<br>"
            f"Risk: {score:.0%} ({risk_label(score)})"
        )

    fig.add_trace(
        go.Scattergeo(
            lon=lons,
            lat=lats,
            text=texts,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=0.5, color="#333333"),
            ),
            hoverinfo="text",
        )
    )

    # Route lines
    for src, tgt in HUB_ROUTES:
        if src not in HUB_GEO or tgt not in HUB_GEO:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=[HUB_GEO[src]["lon"], HUB_GEO[tgt]["lon"]],
                lat=[HUB_GEO[src]["lat"], HUB_GEO[tgt]["lat"]],
                mode="lines",
                line=dict(width=1, color="#888888"),
                hoverinfo="none",
            )
        )

    fig.update_layout(
        geo=dict(
            scope="north america",
            projection_type="azimuthal equal area",
            showland=True,
            landcolor="#111111",
            showcountries=True,
            countrycolor="#444444",
            coastlinecolor="#444444",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


with col_left:
    st.markdown("### Global Network Risk Map")
    if last_cycle:
        fig = build_map_figure(last_cycle["risk_scores"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run at least one tick to see the live risk map.")

    st.markdown("### Hub Risk Table")
    if last_cycle:
        rows = []
        for nid, score in sorted(
            last_cycle["risk_scores"].items(), key=lambda kv: -kv[1]
        ):
            node = env.nodes.get(nid)
            rows.append(
                {
                    "Hub": nid,
                    "Location": getattr(node, "name", nid),
                    "Region": getattr(node, "region", "?"),
                    "Risk": f"{score:.1%}",
                    "Level": risk_label(score),
                    "Backlog": getattr(node, "current_backlog", 0.0),
                    "Health": f"{getattr(node, 'health_status', 1.0):.0%}",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.empty()


with col_right:
    st.markdown("### Agent Terminal (LLM Chain of Thought)")
    terminal_area = st.empty()

    if last_cycle and last_cycle["thought"]:
        t = last_cycle["thought"]
        lines = []
        lines.append(f"**OBSERVE** — {t.observation}")
        lines.append("")
        lines.append("**REASON (LLM internal monologue)**")
        for i, step in enumerate(t.chain_of_thought, 1):
            lines.append(f"{i}. {step}")
        lines.append("")
        lines.append(f"**DECIDE** — {t.decision}")
        lines.append(
            f"**ACTION** — `{t.action.value}`  "
            f"params={t.action_params if t.action_params else '{}'}"
        )
        terminal_area.markdown("\n\n".join(lines))

        # Long-term memory indicator (blue info block)
        if getattr(t, "memory_recall_hit", False):
            st.info("Long-term memory: similar past incidents recalled from ChromaDB.")
        else:
            st.caption("Long-term memory: no strongly similar incidents found this tick.")
    else:
        terminal_area.info("Awaiting first agent cycle...")

    st.markdown("### Guardrail / Action Outcome")
    if last_cycle:
        result = last_cycle["result"]
        guardrail_msg = last_cycle["guardrail_message"]
        blocked = last_cycle["guardrail_triggered"]

        if blocked:
            st.error(guardrail_msg)
        else:
            st.success(guardrail_msg)

        if result:
            st.markdown("**Last tool invocation**")
            st.json(result)
        else:
            st.caption("No tool executed on the last tick.")
    else:
        st.info("No actions taken yet.")

