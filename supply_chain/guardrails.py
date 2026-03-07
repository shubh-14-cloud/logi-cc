"""
supply_chain/guardrails.py
===========================
Safety interception layer between the agent and tool execution.

Policy Rules
------------
  RULE-01  Cost gate    : reroute_cost  > $500   → requires_approval = True
  RULE-02  Volume gate  : rerouted_vol  > 10%    → requires_approval = True
  RULE-03  Escalation   : any `request_human_intervention` call → always flag
  RULE-04  Audit log    : every intercepted command is recorded regardless

Approval Modes
--------------
  auto_approve=False  (default)  — HALT execution; return approved=False.
                                   In production the main loop would await a
                                   human callback before re-submitting.
  auto_approve=True              — Demo/simulation mode; applies a secondary
                                   cost+volume cap to simulate a human operator
                                   who approves "reasonable" escalations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ─── Thresholds ───────────────────────────────────────────────────────────────

COST_APPROVAL_USD    = 500.0    # USD
VOLUME_APPROVAL_PCT  = 0.10     # 10 % of total network volume

# Secondary cap used in auto-approve demo mode
_AUTO_APPROVE_COST_CAP   = 1_200.0   # USD
_AUTO_APPROVE_VOLUME_CAP = 0.22       # 22 %


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """Structured outcome of a single policy evaluation."""

    requires_approval:  bool
    triggered_rules:    List[str]
    cost_estimate_usd:  float
    volume_fraction:    float
    action:             str
    halt_reason:        str     = ""
    approved_by:        str     = ""    # "AUTO-DEMO" | "HUMAN" | ""
    timestamp:          float   = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return not self.requires_approval

    def summary(self) -> str:
        if self.passed:
            return f"PASS — within policy limits (${self.cost_estimate_usd:.0f}, {self.volume_fraction:.1%})"
        return (
            f"BLOCKED — {' | '.join(self.triggered_rules)}  "
            f"[${self.cost_estimate_usd:.0f}, {self.volume_fraction:.1%}]"
        )


@dataclass
class AuditRecord:
    """Immutable audit-log entry for every intercepted command."""
    command:     Dict[str, Any]
    result:      GuardrailResult
    approved:    bool
    mode:        str   # "PASS" | "BLOCKED" | "AUTO-DEMO" | "HUMAN"
    timestamp:   float = field(default_factory=time.time)


# ─── Guardrail Layer ──────────────────────────────────────────────────────────

class GuardrailLayer:
    """
    Policy-enforcement and human-in-the-loop safety layer.

    Parameters
    ----------
    cost_threshold_usd   : float — max reroute cost before requiring approval
    volume_threshold_pct : float — max rerouted-volume fraction before approval
    total_network_volume : float — denominator for volume % calculation
    auto_approve         : bool  — True = demo mode (simulated human approval)
    """

    def __init__(
        self,
        cost_threshold_usd:   float = COST_APPROVAL_USD,
        volume_threshold_pct: float = VOLUME_APPROVAL_PCT,
        total_network_volume: float = 2_500.0,
        auto_approve:         bool  = False,
    ) -> None:
        self.cost_threshold    = cost_threshold_usd
        self.volume_threshold  = volume_threshold_pct
        self.total_net_vol     = max(total_network_volume, 1.0)
        self.auto_approve      = auto_approve

        self.audit_log:        List[AuditRecord] = []
        self.pending_queue:    List[Dict]        = []

    # ── Primary Intercept ─────────────────────────────────────────────────────

    def intercept(self, command: Dict[str, Any]) -> Tuple[bool, GuardrailResult]:
        """
        Evaluate an agent command against all safety policies.

        Parameters
        ----------
        command : dict  with keys  'action', 'params', 'tick'

        Returns
        -------
        (approved: bool, result: GuardrailResult)
        """
        action = command.get("action", "")
        params = command.get("params", {})

        cost_usd       = float(params.get("estimated_cost", params.get("cost_usd", 0.0)))
        volume         = float(params.get("volume", params.get("volume_units", 0.0)))
        volume_frac    = volume / self.total_net_vol

        triggered: List[str] = []

        # RULE-01: Cost gate
        if cost_usd > self.cost_threshold:
            triggered.append(
                f"RULE-01 COST_GATE: ${cost_usd:.2f} exceeds ${self.cost_threshold:.2f} limit"
            )

        # RULE-02: Volume gate
        if volume_frac > self.volume_threshold:
            triggered.append(
                f"RULE-02 VOLUME_GATE: {volume_frac:.1%} exceeds {self.volume_threshold:.0%} limit"
            )

        # RULE-03: Human escalation is always flagged
        if action == "request_human_intervention":
            triggered.append(
                "RULE-03 ESCALATION: Human-intervention tool invoked by agent"
            )

        requires_approval = len(triggered) > 0

        result = GuardrailResult(
            requires_approval = requires_approval,
            triggered_rules   = triggered,
            cost_estimate_usd = cost_usd,
            volume_fraction   = volume_frac,
            action            = action,
            halt_reason       = " | ".join(triggered) if triggered else "OK",
        )

        if not requires_approval:
            # Clean pass — execute immediately
            self._record(command, result, approved=True, mode="PASS")
            return True, result

        # Approval required
        self.pending_queue.append(command)

        if self.auto_approve:
            approved = self._simulate_human_approval(result)
            mode     = "AUTO-DEMO"
            result.approved_by = "AUTO-DEMO"
        else:
            # Real mode: halt and wait for human callback
            approved = False
            mode     = "BLOCKED"

        self._record(command, result, approved=approved, mode=mode)
        if approved and command in self.pending_queue:
            self.pending_queue.remove(command)

        return approved, result

    # ── Human Approval Simulation ─────────────────────────────────────────────

    def _simulate_human_approval(self, result: GuardrailResult) -> bool:
        """
        Simulate a human operator's approval decision in demo mode.

        Approval heuristic:
          • Approve if cost < $1 200 AND volume < 22 %
          • Decline otherwise (forces escalation → incident report)
        """
        cost_ok   = result.cost_estimate_usd <= _AUTO_APPROVE_COST_CAP
        volume_ok = result.volume_fraction   <= _AUTO_APPROVE_VOLUME_CAP
        return cost_ok and volume_ok

    # ── Manual Approval (Human Callback) ──────────────────────────────────────

    def approve_pending(self, index: int = 0) -> bool:
        """
        Human operator approves a queued command.

        In a real system this would be called via a webhook / UI action.
        """
        if index >= len(self.pending_queue):
            return False

        cmd = self.pending_queue.pop(index)
        result = GuardrailResult(
            requires_approval = True,
            triggered_rules   = ["MANUALLY_APPROVED"],
            cost_estimate_usd = float(cmd.get("params", {}).get("estimated_cost", 0)),
            volume_fraction   = 0.0,
            action            = cmd.get("action", ""),
            halt_reason       = "Approved by human operator",
            approved_by       = "HUMAN",
        )
        self._record(cmd, result, approved=True, mode="HUMAN")
        return True

    def decline_pending(self, index: int = 0) -> bool:
        """Human operator explicitly declines a queued command."""
        if index >= len(self.pending_queue):
            return False
        cmd = self.pending_queue.pop(index)
        result = GuardrailResult(
            requires_approval = True,
            triggered_rules   = ["MANUALLY_DECLINED"],
            cost_estimate_usd = 0.0,
            volume_fraction   = 0.0,
            action            = cmd.get("action", ""),
            halt_reason       = "Declined by human operator",
        )
        self._record(cmd, result, approved=False, mode="DECLINED")
        return True

    # ── Audit ─────────────────────────────────────────────────────────────────

    def _record(
        self,
        command:  Dict[str, Any],
        result:   GuardrailResult,
        approved: bool,
        mode:     str,
    ) -> None:
        self.audit_log.append(AuditRecord(
            command  = command,
            result   = result,
            approved = approved,
            mode     = mode,
        ))

    def audit_summary(self) -> Dict[str, Any]:
        """Return aggregate guardrail statistics."""
        total      = len(self.audit_log)
        blocked    = sum(1 for r in self.audit_log if not r.approved)
        auto_ok    = sum(1 for r in self.audit_log if r.mode == "AUTO-DEMO" and r.approved)
        human_ok   = sum(1 for r in self.audit_log if r.mode == "HUMAN")
        rule_counts: Dict[str, int] = {}
        for record in self.audit_log:
            for rule in record.result.triggered_rules:
                rule_key = rule.split(":")[0].strip()
                rule_counts[rule_key] = rule_counts.get(rule_key, 0) + 1

        return {
            "total_intercepted":   total,
            "total_blocked":       blocked,
            "auto_approved":       auto_ok,
            "human_approved":      human_ok,
            "pending":             len(self.pending_queue),
            "rule_hit_counts":     rule_counts,
        }
