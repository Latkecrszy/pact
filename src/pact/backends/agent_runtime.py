"""Provider-neutral runtime helpers for coding-agent subprocesses.

Pact can drive implementation through different CLI agents.  This module
keeps provider-specific command construction in one place so scheduling and
implementation code can reason in terms of capabilities.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ITERATIVE_BACKENDS = {
    "claude_code",
    "claude_code_team",
    "codex_code",
    "codex_code_team",
}

BACKEND_PROVIDERS = {
    "claude_code": "claude",
    "claude_code_team": "claude",
    "codex_code": "codex",
    "codex_code_team": "codex",
}


def is_iterative_backend(name: str) -> bool:
    """Return True when a backend can run an agentic coding shell."""
    return name in ITERATIVE_BACKENDS


def provider_for_backend(name: str) -> str:
    """Map a backend name to its CLI provider."""
    return BACKEND_PROVIDERS.get(name, name)


def normalize_model_for_backend(backend: str, model: str) -> str:
    """Return a model name that is valid for the selected coding shell.

    Pact's global defaults are Claude-oriented.  When a project switches only
    the backend to Codex, inherit Codex's configured default rather than
    passing an invalid Claude model name to ``codex exec``.
    """
    if provider_for_backend(backend) == "codex":
        lowered = model.strip().lower()
        if not lowered or lowered.startswith(("claude", "gemini")):
            return ""
    return model


def require_agent_cli(provider: str) -> str:
    """Return an installed coding-agent CLI or fail with setup guidance."""
    executable = shutil.which(provider)
    if executable:
        return executable
    raise RuntimeError(
        f"{provider} CLI not found on PATH; install and configure {provider} "
        f"before selecting a {provider}_code backend"
    )


def session_id_for_project(project: object) -> str:
    """Resolve the shared runtime session id for worker subprocesses."""
    for key in ("SIGNET_SESSION", "PACT_SESSION_ID"):
        value = os.environ.get(key, "").strip()
        if value:
            return value

    try:
        state = project.load_state()  # type: ignore[attr-defined]
        if state.id:
            return f"pact-{state.id}"
    except Exception:
        pass

    project_dir = Path(getattr(project, "project_dir", Path.cwd())).resolve()
    path_hash = hashlib.sha256(str(project_dir).encode()).hexdigest()[:10]
    return f"pact-{project_dir.name}-{path_hash}"


@dataclass(frozen=True)
class AgentRuntimeSpec:
    """Execution envelope shared by Claude and Codex coding shells."""

    provider: str
    model: str
    working_dir: Path
    session_id: str = ""
    max_turns: int = 30
    sandbox: str = "workspace-write"
    output_file: Path | None = None
    output_schema: Path | None = None

    def env(self) -> dict[str, str]:
        """Environment inherited by worker subprocesses."""
        env = dict(os.environ)
        if self.provider == "claude":
            env.pop("CLAUDECODE", None)
        if self.session_id:
            env["SIGNET_SESSION"] = self.session_id
            env["PACT_SESSION_ID"] = self.session_id
        env["PACT_AGENT_PROVIDER"] = self.provider
        return env


def collaboration_preamble(session_id: str = "") -> str:
    """Prompt preamble that keeps workers on the same memory/safety rails."""
    session_line = (
        f"The shared worker session id is `{session_id}`.\n"
        if session_id else ""
    )
    return (
        "## Pact Runtime Envelope\n\n"
        f"{session_line}"
        "Use Kindex as the durable collaboration layer: search it before "
        "substantial decisions, and capture important discoveries, decisions, "
        "test results, and handoff notes that later workers should know.\n\n"
        "The permission tool is the safety boundary for tool use and preflight "
        "constraints. Do not weaken or bypass it. Treat active preflight red "
        "lines as binding, and stop/report rather than working around them.\n\n"
        "If implementation reveals that a contract or constraint is infeasible, "
        "do not silently deviate. Stop, record the conflict, and reconcile the "
        "Pact artifacts before continuing.\n"
    )


def wrap_worker_prompt(prompt: str, session_id: str = "") -> str:
    """Attach the shared runtime envelope to a worker prompt."""
    return f"{collaboration_preamble(session_id)}\n---\n\n{prompt}"


def build_claude_implement_command(spec: AgentRuntimeSpec) -> list[str]:
    """Command for an iterative Claude Code implementation run."""
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--model",
        spec.model,
        "--max-turns",
        str(spec.max_turns),
        "--allowedTools",
        "Read,Write,Edit,Bash,Glob,Grep",
    ]
    return cmd


def build_codex_exec_command(spec: AgentRuntimeSpec) -> list[str]:
    """Command for an iterative Codex implementation run."""
    cmd = [
        "codex",
        "exec",
        "-C",
        str(spec.working_dir),
        "--sandbox",
        spec.sandbox,
    ]
    if spec.model:
        cmd.extend(["--model", spec.model])
    if spec.output_schema:
        cmd.extend(["--output-schema", str(spec.output_schema)])
    if spec.output_file:
        cmd.extend(["--output-last-message", str(spec.output_file)])
    cmd.append("-")
    return cmd


def build_tmux_agent_command(
    provider: str,
    prompt_file: Path,
    output_file: str,
    model: str,
    max_turns: int = 0,
) -> str:
    """Build a shell command for tmux-based team workers."""
    output_expr = shlex.quote(output_file)
    model_expr = shlex.quote(model)

    if provider == "codex":
        model = normalize_model_for_backend("codex_code", model)
        model_flag = f"--model {shlex.quote(model)} " if model else ""
        cmd = (
            f"codex exec {model_flag}"
            f"--sandbox workspace-write "
            f"--output-last-message {output_expr} "
            f"- < {shlex.quote(str(prompt_file))} "
            f">> {output_expr} 2>&1; "
            f'echo "__CF_AGENT_DONE__" >> {output_expr}'
        )
    else:
        prompt_expr = f"$(cat {shlex.quote(str(prompt_file))})"
        max_turns_flag = (
            f"--max-turns {max_turns} " if max_turns > 0 else ""
        )
        cmd = (
            f"claude -p {prompt_expr} "
            f"--model {model_expr} {max_turns_flag}"
            f"--output-format json "
            f"> {output_expr} 2>&1; "
            f'echo "__CF_AGENT_DONE__" >> {output_expr}'
        )
    return cmd


def build_tmux_launch_command(
    worker_command: str,
    working_dir: str | Path,
    session_id: str = "",
) -> str:
    """Wrap a quoted worker command for tmux's required shell boundary."""
    env_exports = "unset CLAUDECODE; "
    if session_id:
        session_expr = shlex.quote(session_id)
        env_exports += (
            f"export SIGNET_SESSION={session_expr}; "
            f"export PACT_SESSION_ID={session_expr}; "
        )
    return f"cd {shlex.quote(str(working_dir))} && {env_exports}{worker_command}"


def submit_preflight_plan(plan_json: str) -> bool:
    """Submit a preflight plan through a provider-neutral hook.

    Operators can set PACT_PREFLIGHT_SUBMIT_CMD to a command that accepts the
    plan JSON on stdin. If absent, keep the existing Claude MCP fallback.
    """
    configured = os.environ.get("PACT_PREFLIGHT_SUBMIT_CMD", "").strip()
    if configured:
        result = subprocess.run(
            shlex.split(configured),
            input=plan_json,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0

    result = subprocess.run(
        ["claude", "mcp", "call", "signet_preflight_submit", "--", plan_json],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0
