"""Provider-neutral post-implementation review orchestration.

Pact owns the review workflow while Advocate and Simulacrum remain independent
tools. Reports are persisted so either Claude or Codex can process findings,
fix the work, and rerun the same gate.

This is a trusted local-operator tool: target paths, output paths, configured
commands, and installed reviewer executables are chosen by the operator. Pact
does not pass these values through a shell.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def find_simulacrum_command() -> list[str]:
    """Return the explicit override or Pact's packaged Simulacrum runtime."""
    configured = os.environ.get("PACT_SIMULACRUM_CMD", "").strip()
    if configured:
        return shlex.split(configured)
    return [sys.executable, "-m", "pact.vendor.simulacrum"]


def _result(
    *,
    status: str,
    command: list[str] | None = None,
    returncode: int | None = None,
    artifact: Path | None = None,
    stdout: Path | None = None,
    stderr: Path | None = None,
    message: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "command": command or [],
        "returncode": returncode,
        "artifact": str(artifact) if artifact else "",
        "stdout": str(stdout) if stdout else "",
        "stderr": str(stderr) if stderr else "",
        "message": message,
    }


def _run(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_tool_env(),
    )
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    return completed


def _failure_message(stderr_path: Path) -> str:
    """Return a concise subprocess failure message from persisted stderr."""
    try:
        lines = [line.strip() for line in stderr_path.read_text().splitlines()]
    except OSError:
        return ""
    lines = [line for line in lines if line]
    return " ".join(lines[-3:])


def _tool_env() -> dict[str, str]:
    """Build per-process credentials without mutating the caller's exports."""
    env = dict(os.environ)
    fallbacks = {
        "ANTHROPIC_API_KEY": (
            "PACT_REVIEW_ANTHROPIC_API_KEY",
        ),
        "OPENAI_API_KEY": (
            "PACT_REVIEW_OPENAI_API_KEY",
        ),
    }
    for primary, candidates in fallbacks.items():
        if env.get(primary):
            continue
        for candidate in candidates:
            if env.get(candidate):
                env[primary] = env[candidate]
                break
    return env


def select_advocate_provider(explicit: str = "") -> str:
    """Select Advocate's provider from explicit config or standard credentials."""
    provider = explicit or os.environ.get("PACT_ADVOCATE_PROVIDER", "")
    if provider:
        if provider not in {"anthropic", "openai", "gemini"}:
            raise ValueError(f"Unsupported Advocate provider: {provider}")
        return provider

    env = _tool_env()
    if env.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if env.get("OPENAI_API_KEY"):
        return "openai"
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        return "gemini"
    return ""


def _advocate_report_error(artifact: Path) -> str:
    """Return an error when Advocate failed internally or found blockers."""
    if not artifact.exists():
        return "advocate exited successfully but did not write a JSON report"
    try:
        data = json.loads(artifact.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"advocate wrote an unreadable JSON report: {exc}"
    if not isinstance(data, dict):
        return "advocate JSON report must be an object"

    failed_personas = []
    for report in data.get("persona_reports", []):
        summary = str(report.get("summary", "")).strip()
        if report.get("error") or summary.lower().startswith("error:"):
            failed_personas.append(str(report.get("persona", "unknown")))
    if failed_personas:
        return (
            f"{len(failed_personas)} Advocate persona(s) failed: "
            + ", ".join(failed_personas)
        )

    blocking = []
    for report in data.get("persona_reports", []):
        for finding in report.get("findings", []):
            severity = str(finding.get("severity", "")).lower()
            if severity in {"critical", "high"}:
                blocking.append(str(finding.get("title", "untitled finding")))
    if blocking:
        preview = ", ".join(blocking[:3])
        suffix = "..." if len(blocking) > 3 else ""
        return f"{len(blocking)} blocking Advocate finding(s): {preview}{suffix}"
    return ""


def run_reviews(
    target: str | Path,
    *,
    claim: str = "",
    output_dir: str | Path | None = None,
    run_advocate: bool = True,
    run_simulacrum: bool = True,
    advocate_provider: str = "",
    timeout: int = 600,
) -> dict[str, Any]:
    """Run requested review tools and persist a machine-readable report."""
    if timeout <= 0:
        raise ValueError("Review timeout must be greater than zero")

    target_path = Path(target).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Review target does not exist: {target_path}")

    cwd = target_path if target_path.is_dir() else target_path.parent
    if output_dir:
        review_dir = Path(output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        stamp = f"{stamp}-{uuid4().hex[:8]}"
        review_dir = cwd / ".pact" / "reviews" / stamp
    review_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "target": str(target_path),
        "claim": claim,
        "output_dir": str(review_dir),
        "advocate": _result(status="not_requested"),
        "simulacrum": _result(status="not_requested"),
        "ok": False,
    }
    requested_statuses: list[str] = []

    if run_advocate:
        advocate = shutil.which("advocate")
        if not advocate:
            report["advocate"] = _result(
                status="unavailable",
                message=(
                    "advocate executable not found on PATH; install Advocate "
                    "or rerun with --sim-only"
                ),
            )
        else:
            artifact = review_dir / "advocate.json"
            stdout = review_dir / "advocate.stdout.txt"
            stderr = review_dir / "advocate.stderr.txt"
            command = [
                advocate,
                "review",
                str(target_path),
                "--output",
                str(artifact),
                "--no-color",
            ]
            provider = select_advocate_provider(advocate_provider)
            if provider:
                command.extend(["--provider", provider])
            try:
                completed = _run(
                    command,
                    cwd=cwd,
                    stdout_path=stdout,
                    stderr_path=stderr,
                    timeout=timeout,
                )
                artifact_error = (
                    _advocate_report_error(artifact)
                    if completed.returncode == 0 else ""
                )
                passed = completed.returncode == 0 and not artifact_error
                report["advocate"] = _result(
                    status="passed" if passed else "failed",
                    command=command,
                    returncode=completed.returncode,
                    artifact=artifact,
                    stdout=stdout,
                    stderr=stderr,
                    message=artifact_error or (
                        _failure_message(stderr)
                        if completed.returncode != 0 else ""
                    ),
                )
            except subprocess.TimeoutExpired:
                report["advocate"] = _result(
                    status="failed",
                    command=command,
                    artifact=artifact,
                    stdout=stdout,
                    stderr=stderr,
                    message=f"advocate timed out after {timeout}s",
                )
            except OSError as exc:
                report["advocate"] = _result(
                    status="failed",
                    command=command,
                    artifact=artifact,
                    stdout=stdout,
                    stderr=stderr,
                    message=f"advocate could not start: {exc}",
                )
        requested_statuses.append(report["advocate"]["status"])

    if run_simulacrum:
        simulacrum = find_simulacrum_command()
        if not claim.strip():
            report["simulacrum"] = _result(
                status="failed",
                message="--claim is required when running Simulacrum",
            )
        elif not simulacrum:
            report["simulacrum"] = _result(
                status="unavailable",
                message=(
                    "Pact's packaged Simulacrum runtime is unavailable; "
                    "reinstall Pact or set PACT_SIMULACRUM_CMD"
                ),
            )
        else:
            artifact = review_dir / "simulacrum.md"
            stderr = review_dir / "simulacrum.stderr.txt"
            command = [*simulacrum, "--quiet", claim]
            try:
                completed = _run(
                    command,
                    cwd=cwd,
                    stdout_path=artifact,
                    stderr_path=stderr,
                    timeout=timeout,
                )
                report["simulacrum"] = _result(
                    status="completed" if completed.returncode == 0 else "failed",
                    command=command,
                    returncode=completed.returncode,
                    artifact=artifact,
                    stdout=artifact,
                    stderr=stderr,
                    message=(
                        _failure_message(stderr)
                        if completed.returncode != 0 else ""
                    ),
                )
            except subprocess.TimeoutExpired:
                report["simulacrum"] = _result(
                    status="failed",
                    command=command,
                    artifact=artifact,
                    stderr=stderr,
                    message=f"Simulacrum timed out after {timeout}s",
                )
            except OSError as exc:
                report["simulacrum"] = _result(
                    status="failed",
                    command=command,
                    artifact=artifact,
                    stderr=stderr,
                    message=f"Simulacrum could not start: {exc}",
                )
        requested_statuses.append(report["simulacrum"]["status"])

    report["ok"] = bool(requested_statuses) and all(
        status in {"passed", "completed"} for status in requested_statuses
    )
    (review_dir / "review.json").write_text(json.dumps(report, indent=2))
    return report


def render_review_summary(report: dict[str, Any]) -> str:
    """Render a concise human-readable review summary."""
    lines = [
        f"Review tools: {'completed' if report.get('ok') else 'needs attention'}",
        f"  Target: {report.get('target', '')}",
        f"  Reports: {report.get('output_dir', '')}",
    ]
    for name in ("advocate", "simulacrum"):
        result = report.get(name, {})
        status = result.get("status", "unknown")
        message = result.get("message", "")
        suffix = f" - {message}" if message else ""
        lines.append(f"  {name.capitalize()}: {status}{suffix}")
    return "\n".join(lines)
