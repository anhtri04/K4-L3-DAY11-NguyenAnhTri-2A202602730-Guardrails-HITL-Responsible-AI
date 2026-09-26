"""
Interactive Rich CLI to chat with Blue / Red / Red Advance and watch guardrails live.

Run from repo root:
    python src/main.py --chat blue          # defended agent (your CP2-3 plugins)
    python src/main.py --chat red           # soft target (leaks by design)
    python src/main.py --chat red_advance   # hard target (strong guardrails)

Features: multi-turn sliding-window context (default 20 messages), per-turn
verdict (ALLOWED / BLOCKED + layer / LEAKED), live observability panel
(requests, blocks, leaks, rate-limit hits, alerts), /attack N shortcuts to
fire the 5 CP4 adversarial prompts, /save to export the session audit.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()

COMMANDS = "/help /stats /history /reset /attack <1-5> /save /quit"


def _target_name(mode: str) -> str:
    return {"blue": "blue", "red": "red_default", "red_advance": "red_advance"}[mode]


def build_session(mode: str, window: int) -> dict:
    """Create agent + runner + observability for a chat mode."""
    from assignment.pipeline import build_production_plugins, build_observability
    from core.config import blue_provider_label, red_provider_label

    if mode == "blue":
        from agents.agent import create_blue_agent

        plugins = build_production_plugins(use_llm_judge=False)
        audit, monitor = build_observability()
        agent, runner = create_blue_agent(plugins)
        label = f"Blue [{blue_provider_label()}] + your guardrails"
    elif mode == "red":
        from agents.agent import create_red_agent_default
        from assignment.audit_log import AuditLogPlugin
        from assignment.monitoring import MonitoringAlert

        plugins, audit, monitor = [], AuditLogPlugin(), MonitoringAlert()
        agent, runner = create_red_agent_default()
        label = f"Red [{red_provider_label('default')}] — no guardrails, leaks by design"
    else:
        from agents.guards_agent import create_red_agent_advance
        from assignment.audit_log import AuditLogPlugin
        from assignment.monitoring import MonitoringAlert

        plugins, audit, monitor = [], AuditLogPlugin(), MonitoringAlert()
        agent, runner = create_red_agent_advance()
        label = f"Red Advance [{red_provider_label('advance')}] — strong guardrails"
    return {
        "mode": mode, "label": label, "agent": agent, "runner": runner,
        "plugins": plugins, "audit": audit, "monitor": monitor,
        "history": deque(maxlen=window), "window": window,
        "turns": 0, "blocked": 0, "leaked": 0, "refused": 0, "errors": 0,
        "last_layer": None, "transcript": [],
    }


def _plugin_snapshot(plugins: list) -> dict:
    snap = {}
    for p in plugins:
        name = getattr(p, "name", type(p).__name__)
        snap[name] = {
            k: getattr(p, k, 0)
            for k in ("total_count", "blocked_count", "redacted_count")
        }
    return snap


def classify_blue_turn(plugins, before: dict, reply: str) -> tuple[bool, str | None]:
    """Attribute a Blue reply to a layer via plugin-counter deltas."""
    after = _plugin_snapshot(plugins)
    rl = after.get("rate_limiter", {})
    if rl.get("blocked_count", 0) > before.get("rate_limiter", {}).get("blocked_count", 0):
        return True, "rate_limiter"
    ig = after.get("input_guardrail", {})
    if ig.get("blocked_count", 0) > before.get("input_guardrail", {}).get("blocked_count", 0):
        return True, "input_guardrail"
    og = after.get("output_guardrail", {})
    b0 = before.get("output_guardrail", {})
    if og.get("blocked_count", 0) > b0.get("blocked_count", 0):
        return True, "output_guardrail"
    if og.get("redacted_count", 0) > b0.get("redacted_count", 0):
        return True, "output_guardrail(redacted)"
    return False, None


def stats_table(sess: dict) -> Table:
    m = sess["monitor"]
    snap = m.snapshot()
    t = Table(title=f"Observability — {sess['mode']}", show_header=True)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Turns", str(sess["turns"]))
    t.add_row("Blocked (plugin)", str(sess["blocked"]))
    t.add_row("Block rate", f"{snap['block_rate']:.0%}")
    t.add_row("Rate-limit hits", str(snap["rate_limit_hits"]))
    t.add_row("Leaked responses", f"[red]{sess['leaked']}[/red]" if sess["leaked"] else "0")
    t.add_row("Model refusals", str(sess["refused"]))
    t.add_row("Context window", f"{len(sess['history'])}/{sess['window']} msgs")
    t.add_row("Last layer", str(sess["last_layer"]))
    for a in snap["alerts"][-3:]:
        t.add_row(f"ALERT {a['metric']}", f"{a['value']} > {a['threshold']}")
    plugs = _plugin_snapshot(sess["plugins"])
    for name, c in plugs.items():
        t.add_row(f"plugin:{name}", f"total={c['total_count']} blocked={c['blocked_count']}")
    return t


async def do_turn(sess: dict, text: str) -> None:
    """Send one message, classify the outcome, update observability, render."""
    from attacks.attacks import classify_attack_outcome, response_leaked_secrets
    from agents.guards_agent import check_secret_leak
    from core.utils import chat_with_agent

    mode = sess["mode"]
    audit, monitor = sess["audit"], sess["monitor"]
    rid = f"chat:{sess['turns']}"
    audit.record_input(user_id="chat-user", text=text, request_id=rid)
    before = _plugin_snapshot(sess["plugins"])

    try:
        reply, _ = await chat_with_agent(
            sess["agent"], sess["runner"], text, history=list(sess["history"])
        )
        reply = reply or "(empty reply)"
        error = None
    except Exception as e:  # e.g. Blue model slug has no live endpoint
        reply, error = f"(LLM error: {type(e).__name__}: {str(e)[:200]})", True

    if mode == "blue" and not error:
        blocked, layer = classify_blue_turn(sess["plugins"], before, reply)
        leaked = check_secret_leak(reply)
        refused = (not blocked and not leaked and "cannot" in reply.lower()
                   and len(reply) < 300)
    elif not error:
        out = classify_attack_outcome(text, reply, target_name=_target_name(mode))
        blocked, layer = out["blocked"], out["layer"]
        leaked = out["leaked"]
        refused = layer == "model_refuse"
    else:
        blocked, layer, leaked, refused = False, "error", False, False

    sess["turns"] += 1
    sess["blocked"] += bool(blocked)
    sess["leaked"] += bool(leaked)
    sess["refused"] += bool(refused)
    sess["errors"] += bool(error)
    sess["last_layer"] = layer
    monitor.total_requests += 1
    monitor.blocked_requests += bool(blocked)
    if layer == "rate_limiter":
        monitor.rate_limit_hits += 1
    monitor.check_metrics()
    audit.record_output(user_id="chat-user", text=reply, blocked=blocked,
                        layer=layer, request_id=rid)
    if not error:
        sess["history"].append({"role": "user", "content": text})
        sess["history"].append({"role": "assistant", "content": reply})
    sess["transcript"].append({"turn": sess["turns"], "input": text,
                               "reply": reply, "blocked": blocked,
                               "layer": layer, "leaked": leaked})

    if leaked:
        verdict = "[bold red]LEAKED — response contains a protected secret[/bold red]"
    elif blocked:
        verdict = f"[bold yellow]BLOCKED @ {layer}[/bold yellow]"
    elif error:
        verdict = "[red]ERROR (see reply)[/red]"
    elif refused:
        verdict = "[cyan]MODEL REFUSE (no plugin fired)[/cyan]"
    else:
        verdict = "[green]ALLOWED[/green]"
    console.print(Panel(reply, title=f"[bold]{mode}[/bold] reply · {verdict}",
                        border_style="red" if (leaked or error) else "yellow" if blocked else "green"))
    console.print(stats_table(sess))


def show_help() -> None:
    console.print(Panel(
        "Type banking questions or jailbreaks — guardrails are evaluated live.\n"
        f"Commands: {COMMANDS}\n"
        "/attack N fires CP4 adversarial prompt N (1-5) for quick guardrail tests.",
        title="Help", border_style="cyan"))


def save_session(sess: dict) -> Path:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / f"chat_session_{sess['mode']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": sess["mode"], "label": sess["label"],
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"turns": sess["turns"], "blocked": sess["blocked"],
                    "leaked": sess["leaked"], "refused": sess["refused"],
                    "errors": sess["errors"],
                    "monitoring": sess["monitor"].snapshot()},
        "transcript": sess["transcript"],
        "audit": sess["audit"].logs,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


async def run_chat(mode: str, window: int = 20) -> None:
    from core.config import setup_api_key

    setup_api_key()
    sess = build_session(mode, window)
    console.print(Panel(f"[bold]{sess['label']}[/bold]\nContext window: last {window} messages · {COMMANDS}",
                        title="Day 11 — Guardrail Chat Demo", border_style="magenta"))
    try:
        from attacks.attacks import adversarial_prompts
        console.print(f"[dim]Loaded {len(adversarial_prompts)} CP4 attack prompts — try /attack 1[/dim]")
    except Exception:
        pass

    while True:
        try:
            text = Prompt.ask("\n[bold cyan]you[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Session ended.[/yellow]")
            break
        if not text:
            continue
        if text.startswith("/"):
            cmd, *rest = text.split()
            if cmd in ("/quit", "/exit"):
                console.print("[yellow]Session ended.[/yellow]")
                break
            elif cmd == "/help":
                show_help()
            elif cmd == "/stats":
                console.print(stats_table(sess))
            elif cmd == "/history":
                for m in sess["history"]:
                    who = "you" if m["role"] == "user" else sess["mode"]
                    console.print(f"[dim]{who}:[/dim] {m['content'][:160]}")
            elif cmd == "/reset":
                sess["history"].clear()
                for p in sess["plugins"]:
                    if getattr(p, "name", "") == "rate_limiter":
                        p.user_windows.clear()
                console.print("[yellow]Context window + rate-limit windows cleared.[/yellow]")
            elif cmd == "/attack":
                try:
                    from attacks.attacks import adversarial_prompts
                    n = int(rest[0]) if rest else 1
                    p = next(a for a in adversarial_prompts if a["id"] == n)
                    console.print(f"[magenta]Firing attack #{n} ({p['category']})[/magenta]")
                    await do_turn(sess, p["input"])
                except (ValueError, StopIteration):
                    console.print("[red]Usage: /attack <1-5>[/red]")
            elif cmd == "/save":
                console.print(f"[green]Saved → {save_session(sess)}[/green]")
            else:
                console.print(f"[red]Unknown command. {COMMANDS}[/red]")
            continue
        await do_turn(sess, text)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Day 11 Rich chat demo (guardrail testing)")
    ap.add_argument("--mode", choices=["blue", "red", "red_advance"], required=True)
    ap.add_argument("--window", type=int, default=20, help="sliding-window context size in messages")
    args = ap.parse_args()
    asyncio.run(run_chat(args.mode, max(2, args.window)))


if __name__ == "__main__":
    main()
