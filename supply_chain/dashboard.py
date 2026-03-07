"""
supply_chain/dashboard.py
==========================
Rich-based CLI live dashboard for the Agentic Supply Chain Layer.
Windows-safe: all output is ASCII-only; Rich console forced to utf-8 mode.
"""

from __future__ import annotations

import io
import os
import sys
from typing import Dict, List, Optional, Tuple

# Force utf-8 on Windows stdout so Rich and print() both work
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Rich availability guard ────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich.columns import Columns
    from rich.rule    import Rule
    from rich         import box
    _RICH = True
except ImportError:
    _RICH = False


# ─── Risk-level styling helpers ───────────────────────────────────────────────

_RISK_BANDS: List[Tuple[float, str, str, str]] = [
    # (threshold, label, rich_style, ascii_prefix)
    (0.82, "CRITICAL", "bold red",      "!!!"),
    (0.65, "HIGH",     "red",           "!! "),
    (0.45, "MEDIUM",   "yellow",        "!  "),
    (0.25, "LOW",      "bright_yellow", "-  "),
    (0.00, "NOMINAL",  "bright_green",  "   "),
]


def _classify(score: float) -> Tuple[str, str, str]:
    for thresh, label, style, prefix in _RISK_BANDS:
        if score >= thresh:
            return label, style, prefix
    return "NOMINAL", "bright_green", "   "


def _bar(score: float, width: int = 22) -> str:
    filled = round(score * width)
    return "#" * filled + "-" * (width - filled)


def _score_text(score: float) -> str:
    return f"{score:5.1%}"


# ─── Dashboard ────────────────────────────────────────────────────────────────

class SupplyChainDashboard:
    def __init__(self, max_log_lines: int = 10) -> None:
        if _RICH:
            # force_terminal prevents Rich falling back to legacy Windows renderer
            self.console = Console(
                highlight=False,
                force_terminal=True,
                legacy_windows=False,
                file=sys.stdout,
            )
        else:
            self.console = None
        self.event_log: List[str] = []
        self.max_log_lines = max_log_lines

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        tick:              int,
        risk_scores:       Dict[str, float],
        node_names:        Dict[str, str],
        thought_step,
        action_result:     Optional[Dict],
        guardrail_message: str,
        guardrail_blocked: bool,
        model_params:      int,
        failure_node:      Optional[str] = None,
    ) -> None:
        if _RICH:
            self._render_rich(
                tick, risk_scores, node_names, thought_step,
                action_result, guardrail_message, guardrail_blocked,
                model_params, failure_node,
            )
        else:
            self._render_plain(
                tick, risk_scores, node_names, thought_step,
                action_result, guardrail_message, guardrail_blocked,
            )

    # ── Rich renderer (ASCII symbols only) ───────────────────────────────────

    def _render_rich(
        self, tick, risk_scores, node_names, thought_step,
        action_result, guardrail_message, guardrail_blocked,
        model_params, failure_node,
    ) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        c = self.console

        # Header
        peak_node  = max(risk_scores, key=risk_scores.get) if risk_scores else "-"
        peak_score = risk_scores.get(peak_node, 0.0)
        _, peak_style, _ = _classify(peak_score)

        c.print(Rule(
            f"[bold cyan] AGENTIC SUPPLY CHAIN LAYER  |  "
            f"Tick {tick:04d}  |  ST-GAT {model_params:,} params  |  "
            f"Peak Risk [{peak_style}]{peak_score:.0%}[/{peak_style}]"
            f" @ {node_names.get(peak_node, peak_node)} [/bold cyan]",
            style="cyan",
        ))

        # Risk score table
        t = Table(
            title        = "[bold white] GNN Blast-Radius Risk Scores",
            box          = box.SIMPLE_HEAVY,
            header_style = "bold magenta",
            show_lines   = False,
            expand       = False,
        )
        t.add_column("Hub ID",   style="cyan dim", width=10)
        t.add_column("Location", style="white",    width=16)
        t.add_column("Risk Bar",                   width=26)
        t.add_column("Score",  justify="right",    width=7)
        t.add_column("Level",  justify="center",   width=10)
        t.add_column("Status", justify="center",   width=16)

        for nid, score in sorted(risk_scores.items(), key=lambda kv: -kv[1]):
            label, style, _ = _classify(score)
            is_failed = (nid == failure_node)

            bar_txt   = f"[{style}]{_bar(score)}[/{style}]"
            score_txt = f"[{style}]{_score_text(score)}[/{style}]"
            label_txt = f"[{style}]{label}[/{style}]"

            if is_failed:
                status = "[bold red]** FAILURE **[/bold red]"
            elif score >= 0.65:
                status = "[red]/\\ CASCADING[/red]"
            elif score >= 0.45:
                status = "[yellow]^ ELEVATED[/yellow]"
            else:
                status = "[bright_green]* NOMINAL[/bright_green]"

            t.add_row(nid, node_names.get(nid, nid), bar_txt, score_txt, label_txt, status)

        c.print(t)
        c.print()

        # Agent monologue
        if thought_step:
            cot_lines = "\n".join(
                f"  [dim white]{i+1}.[/dim white] {step}"
                for i, step in enumerate(thought_step.chain_of_thought)
            )
            arrow = "->"
            monologue = (
                f"[bright_cyan]OBSERVE[/bright_cyan]  {thought_step.observation}\n\n"
                f"[yellow]REASON [/yellow]\n{cot_lines}\n\n"
                f"[bright_green]DECIDE [/bright_green]  {thought_step.decision}\n\n"
                f"[bold white]ACTION [/bold white]  "
                f"[bold magenta]{thought_step.action.value.upper()}[/bold magenta]"
                + (
                    f"  {arrow}  [dim]{thought_step.action_params}[/dim]"
                    if thought_step.action_params else ""
                )
            )
            c.print(Panel(
                monologue,
                title        = "[bold yellow] Agent Internal Monologue  (ReAct Loop)",
                border_style = "yellow",
                padding      = (0, 1),
            ))
            c.print()

        # Event log
        if action_result:
            msg  = action_result.get("message", str(action_result))
            tool = action_result.get("tool", "?")
            ok   = action_result.get("success", True)
            icon = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
            self.event_log.append(f"[Tick {tick:03d}] [{icon}] [{tool.upper()}] {msg}")
            if len(self.event_log) > self.max_log_lines:
                self.event_log.pop(0)

        log_body = "\n".join(
            f"  {'[dim]' if i < len(self.event_log) - 1 else '[white bold]'}"
            f"{line}"
            f"{'[/dim]' if i < len(self.event_log) - 1 else '[/white bold]'}"
            for i, line in enumerate(self.event_log)
        ) or "  [dim]No actions logged yet...[/dim]"

        guard_style = "bold red" if guardrail_blocked else "bold green"
        guard_icon  = "[X] BLOCKED" if guardrail_blocked else "[v] PASS"
        guard_body  = f"[{guard_style}]{guard_icon}[/{guard_style}]\n\n{guardrail_message}"

        c.print(Columns(
            [
                Panel(log_body,   title="[bold blue] Action Event Log",     border_style="blue"),
                Panel(guard_body, title="[bold] Guardrail / Safety Layer",  border_style=guard_style.split()[-1]),
            ],
            expand=True,
        ))

        c.print(Rule("[dim]Ctrl-C to stop  |  --auto-approve flag controls guardrail mode[/dim]"))

    # ── Plain-text fallback ───────────────────────────────────────────────────

    def _render_plain(
        self, tick, risk_scores, node_names, thought_step,
        action_result, guardrail_message, guardrail_blocked,
    ) -> None:
        W = 80
        print("\n" + "=" * W)
        print(f"  AGENTIC SUPPLY CHAIN LAYER  |  Tick {tick:04d}")
        print("=" * W)

        print("\n  GNN BLAST-RADIUS RISK SCORES")
        print("  " + "-" * (W - 2))
        for nid, score in sorted(risk_scores.items(), key=lambda kv: -kv[1]):
            label, _, prefix = _classify(score)
            name = node_names.get(nid, nid)
            bar  = _bar(score, width=18)
            print(f"  {prefix} {nid:10s}  {name:16s}  [{bar}]  {_score_text(score)}  {label}")

        if thought_step:
            print(f"\n  AGENT MONOLOGUE")
            print("  " + "-" * (W - 2))
            print(f"  OBS:    {thought_step.observation}")
            for i, step in enumerate(thought_step.chain_of_thought, 1):
                print(f"  [{i}]     {step}")
            print(f"  DECIDE: {thought_step.decision}")
            print(f"  ACTION: {thought_step.action.value.upper()}")
            if thought_step.action_params:
                print(f"          {thought_step.action_params}")

        if action_result:
            msg = action_result.get("message", "")
            print(f"\n  RESULT: {msg}")
            self.event_log.append(f"[{tick:03d}] {msg}")
            if len(self.event_log) > self.max_log_lines:
                self.event_log.pop(0)

        guard_prefix = "!!! " if guardrail_blocked else "    "
        print(f"\n  GUARDRAIL: {guard_prefix}{guardrail_message}")
        print("=" * W)
