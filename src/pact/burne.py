"""BURN-E constrained agent commands.

These commands are intentionally narrower than Pact's general-purpose
pipeline commands. They are designed for safe-env repair lanes where the
workspace, component, Pact project, source roots, and spend caps are supplied
by a trusted caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_WALL_SECONDS = 900
MAX_MODEL_TOKENS = 50_000
MAX_TOOL_CALLS = 75
MAX_USD_CENTS = 100
REPORT_SCHEMA_VERSION = "pact-burne-agent/v1"

CAP_ENV = {
    "max_wall_seconds": "AGENT_SAFE_AGENT_MAX_WALL_SECONDS",
    "max_model_tokens": "AGENT_SAFE_AGENT_MAX_MODEL_TOKENS",
    "max_tool_calls": "AGENT_SAFE_AGENT_MAX_TOOL_CALLS",
    "max_usd": "AGENT_SAFE_AGENT_MAX_USD",
}

FORBIDDEN_REPAIR_FORBIDDEN_WRITES = {
    "contracts",
    "visible-tests",
    "control-plane",
    "hidden-oracle",
}

FORBIDDEN_SOURCE_ROOT_TOP_LEVELS = {
    ".github",
    "agent-safe",
    "contracts",
    "hidden-oracle",
    "manifests",
    "pact",
}

COMPONENT_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


@dataclass(frozen=True)
class ResolvedPath:
    relative: str
    resolved: Path


@dataclass(frozen=True)
class AgentResult:
    status: str
    exit_code: int
    stdout: str
    stderr: str
    output: str
    elapsed_seconds: float


def violation(name: str, message: str) -> dict[str, str]:
    return {"name": name, "message": message}


def _short_text(value: object, limit: int = 4000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[-limit:]


def _parse_positive_int_cap(
    env: Mapping[str, str],
    key: str,
    maximum: int,
    violations: list[dict[str, str]],
) -> int:
    raw = str(env.get(key, "") or "").strip()
    if not raw:
        violations.append(violation(key, f"{key} is required"))
        return 0
    if not re.fullmatch(r"[1-9][0-9]*", raw):
        violations.append(violation(key, f"{key} must be a positive integer"))
        return 0
    value = int(raw, 10)
    if value > maximum:
        violations.append(violation(key, f"{key} must be <= {maximum}"))
    return value


def _parse_usd_cap(env: Mapping[str, str], violations: list[dict[str, str]]) -> tuple[int, str]:
    key = CAP_ENV["max_usd"]
    raw = str(env.get(key, "") or "").strip()
    if not raw:
        violations.append(violation(key, f"{key} is required"))
        return 0, "0.00"
    if not re.fullmatch(r"[0-9]+(\.[0-9]{1,2})?", raw):
        violations.append(violation(key, f"{key} must be a USD decimal with at most two fractional digits"))
        return 0, "0.00"
    dollars = raw.split(".", 1)[0]
    cents = "00"
    if "." in raw:
        cents = raw.split(".", 1)[1].ljust(2, "0")
    total_cents = int(dollars, 10) * 100 + int(cents, 10)
    if total_cents <= 0 or total_cents > MAX_USD_CENTS:
        violations.append(violation(key, f"{key} must be > 0.00 and <= 1.00"))
    return total_cents, f"{total_cents // 100}.{total_cents % 100:02d}"


def parse_caps(env: Mapping[str, str], violations: list[dict[str, str]]) -> dict[str, object]:
    usd_cents, usd = _parse_usd_cap(env, violations)
    return {
        "max_wall_seconds": _parse_positive_int_cap(
            env, CAP_ENV["max_wall_seconds"], MAX_WALL_SECONDS, violations,
        ),
        "max_model_tokens": _parse_positive_int_cap(
            env, CAP_ENV["max_model_tokens"], MAX_MODEL_TOKENS, violations,
        ),
        "max_tool_calls": _parse_positive_int_cap(
            env, CAP_ENV["max_tool_calls"], MAX_TOOL_CALLS, violations,
        ),
        "max_usd": usd,
        "max_usd_cents": usd_cents,
    }


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _clean_relative_path(
    value: str,
    *,
    field_name: str,
    cwd: Path,
    violations: list[dict[str, str]],
    require_exists: bool,
    require_dir: bool,
) -> ResolvedPath | None:
    raw = str(value or "").strip()
    if not raw:
        violations.append(violation(field_name, f"{field_name} is required"))
        return None
    if raw in {".", "./"}:
        violations.append(violation(field_name, f"{field_name} must not be the workspace root"))
        return None
    if raw.startswith("-"):
        violations.append(violation(field_name, f"{field_name} must not look like a command flag"))
        return None
    if raw.startswith("~") or raw.startswith("/") or "//" in raw:
        violations.append(violation(field_name, f"{field_name} must be a normalized relative path"))
        return None

    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or "~" in path.parts:
        violations.append(violation(field_name, f"{field_name} must be a normalized relative path"))
        return None

    root = cwd.resolve()
    resolved = (root / path).resolve()
    if not _path_is_relative_to(resolved, root):
        violations.append(violation(field_name, f"{field_name} must stay inside the current workspace"))
        return None
    if require_exists and not resolved.exists():
        violations.append(violation(field_name, f"{field_name} does not exist in the current workspace"))
        return None
    if require_dir and resolved.exists() and not resolved.is_dir():
        violations.append(violation(field_name, f"{field_name} must be a directory"))
        return None
    return ResolvedPath(path.as_posix(), resolved)


def _validate_component(env: Mapping[str, str], violations: list[dict[str, str]]) -> str:
    component = str(env.get("AGENT_SAFE_COMPONENT", "") or "").strip()
    if not component:
        violations.append(violation("AGENT_SAFE_COMPONENT", "AGENT_SAFE_COMPONENT is required"))
    elif not COMPONENT_RE.fullmatch(component):
        violations.append(violation("AGENT_SAFE_COMPONENT", "AGENT_SAFE_COMPONENT has an invalid value"))
    return component


def _validate_pact_project(
    env: Mapping[str, str],
    cwd: Path,
    component: str,
    violations: list[dict[str, str]],
) -> ResolvedPath | None:
    project = _clean_relative_path(
        str(env.get("AGENT_SAFE_PACT_PROJECT", "") or ""),
        field_name="AGENT_SAFE_PACT_PROJECT",
        cwd=cwd,
        violations=violations,
        require_exists=True,
        require_dir=True,
    )
    if project is None:
        return None

    parts = Path(project.relative).parts
    if len(parts) < 2:
        violations.append(violation("AGENT_SAFE_PACT_PROJECT", "Pact project must be component-scoped, not a top-level root"))
    allowed_names = {component}
    pact_component_id = str(env.get("AGENT_SAFE_PACT_COMPONENT_ID", "") or "").strip()
    if pact_component_id:
        allowed_names.add(pact_component_id)
    if parts[-1] not in allowed_names:
        violations.append(
            violation(
                "AGENT_SAFE_PACT_PROJECT",
                "Pact project directory name must match AGENT_SAFE_COMPONENT or AGENT_SAFE_PACT_COMPONENT_ID",
            )
        )
    return project


def _validate_output_path(
    value: str,
    cwd: Path,
    violations: list[dict[str, str]],
) -> Path | None:
    if not value:
        return None
    output = _clean_relative_path(
        value,
        field_name="output",
        cwd=cwd,
        violations=violations,
        require_exists=False,
        require_dir=False,
    )
    if output is None:
        return None
    if output.resolved.exists() and output.resolved.is_dir():
        violations.append(violation("output", "output must be a file path"))
        return None
    return output.resolved


def _validate_source_roots(
    roots: Sequence[str],
    cwd: Path,
    pact_project: ResolvedPath | None,
    violations: list[dict[str, str]],
) -> list[ResolvedPath]:
    if not roots:
        violations.append(violation("source-root", "repair command requires at least one --source-root"))
        return []

    resolved_roots: list[ResolvedPath] = []
    pact_resolved = pact_project.resolved if pact_project else None
    for value in roots:
        root = _clean_relative_path(
            value,
            field_name="source-root",
            cwd=cwd,
            violations=violations,
            require_exists=True,
            require_dir=True,
        )
        if root is None:
            continue
        top = Path(root.relative).parts[0]
        if top in FORBIDDEN_SOURCE_ROOT_TOP_LEVELS:
            violations.append(violation("source-root", f"source root {root.relative} is forbidden"))
        if pact_resolved and (
            _path_is_relative_to(root.resolved, pact_resolved)
            or _path_is_relative_to(pact_resolved, root.resolved)
        ):
            violations.append(violation("source-root", "source root must not overlap the Pact project"))
        resolved_roots.append(root)
    return resolved_roots


def _validate_repair_allowed_context(env: Mapping[str, str], violations: list[dict[str, str]]) -> dict[str, object]:
    raw = str(env.get("AGENT_SAFE_REPAIR_AGENT_ALLOWED_CONTEXT", "") or "").strip()
    if not raw:
        violations.append(violation("AGENT_SAFE_REPAIR_AGENT_ALLOWED_CONTEXT", "repair allowed context is required"))
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        violations.append(violation("AGENT_SAFE_REPAIR_AGENT_ALLOWED_CONTEXT", f"repair allowed context must be JSON: {error}"))
        return {}
    if not isinstance(payload, dict):
        violations.append(violation("AGENT_SAFE_REPAIR_AGENT_ALLOWED_CONTEXT", "repair allowed context must be a JSON object"))
        return {}
    return payload


def _validate_repair_forbidden_writes(env: Mapping[str, str], violations: list[dict[str, str]]) -> list[str]:
    raw = str(env.get("AGENT_SAFE_REPAIR_AGENT_FORBIDDEN_WRITES", "") or "")
    values = {item.strip() for item in raw.split(",") if item.strip()}
    missing = sorted(FORBIDDEN_REPAIR_FORBIDDEN_WRITES - values)
    if missing:
        violations.append(
            violation(
                "AGENT_SAFE_REPAIR_AGENT_FORBIDDEN_WRITES",
                f"repair forbidden writes must include: {', '.join(missing)}",
            )
        )
    return sorted(values)


def _file_fingerprint(path: Path) -> str:
    if path.is_symlink():
        return "symlink:" + os.readlink(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def workspace_manifest(cwd: Path) -> dict[str, str]:
    root = cwd.resolve()
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts:
            continue
        if not path.is_file() and not path.is_symlink():
            continue
        resolved = path.resolve()
        if not _path_is_relative_to(resolved, root) and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            manifest[relative] = _file_fingerprint(path)
        except OSError as error:
            manifest[relative] = f"unreadable:{type(error).__name__}"
    return manifest


def _path_allowed(relative: str, allowed_roots: Sequence[str]) -> bool:
    return any(relative == root or relative.startswith(f"{root}/") for root in allowed_roots)


def changed_paths_outside_roots(
    before: Mapping[str, str],
    after: Mapping[str, str],
    allowed_roots: Sequence[str],
) -> list[str]:
    changed: list[str] = []
    for relative in sorted(set(before) | set(after)):
        if before.get(relative) == after.get(relative):
            continue
        if not _path_allowed(relative, allowed_roots):
            changed.append(relative)
    return changed


def _codex_command(cwd: Path, output_file: Path) -> list[str]:
    return [
        "codex",
        "exec",
        "-C",
        str(cwd),
        "--sandbox",
        "workspace-write",
        "--output-last-message",
        str(output_file),
        "-",
    ]


def run_codex_once(cwd: Path, prompt: str, timeout_seconds: int) -> tuple[list[str], AgentResult]:
    with tempfile.TemporaryDirectory(prefix="pact-burne-codex-") as tmp:
        output_file = Path(tmp) / "last-message.txt"
        command = _codex_command(cwd, output_file)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=str(cwd),
                env=dict(os.environ),
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            elapsed = time.monotonic() - started
            return command, AgentResult(
                status="timeout",
                exit_code=124,
                stdout=_short_text(error.stdout),
                stderr=_short_text(error.stderr),
                output="",
                elapsed_seconds=elapsed,
            )

        elapsed = time.monotonic() - started
        output = output_file.read_text(encoding="utf-8") if output_file.exists() else proc.stdout
        return command, AgentResult(
            status="passed" if proc.returncode == 0 else "failed",
            exit_code=int(proc.returncode),
            stdout=_short_text(proc.stdout),
            stderr=_short_text(proc.stderr),
            output=_short_text(output),
            elapsed_seconds=elapsed,
        )


def _base_report(mode: str, cwd: Path, caps: Mapping[str, object], component: str, pact_project: ResolvedPath | None) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": mode,
        "status": "not-run",
        "accepted": False,
        "workspace": str(cwd),
        "component": component,
        "pact_project": pact_project.relative if pact_project else "",
        "source_roots": [],
        "caps": dict(caps),
        "policy": {"passed": False, "violations": []},
        "allowed_write_roots": [],
        "write_guard": {"status": "not-run", "forbidden_changed_paths": []},
        "agent": {
            "provider": "codex",
            "command": [],
            "status": "not-run",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "output": "",
            "elapsed_seconds": 0.0,
        },
    }


def _emit_report(report: Mapping[str, object], output_path: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path is None:
        print(text, end="")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _spec_prompt(
    *,
    component: str,
    pact_project: str,
    caps: Mapping[str, object],
    allowed_roots: Sequence[str],
) -> str:
    return (
        "You are the Pact BURN-E spec-author agent.\n\n"
        "Hard rules:\n"
        f"- Component: {component}\n"
        f"- Pact project: {pact_project}\n"
        f"- Allowed write roots: {', '.join(allowed_roots)}\n"
        "- Do not edit implementation source, workflow files, control-plane files, hidden-oracle files, or agent-safe files.\n"
        "- Do not run broad workspace scans. Inspect only the Pact project and relevant agent artifact inputs.\n"
        "- If the task requires broader context, stop and explain that the allowed context is insufficient.\n\n"
        "Spend caps supplied by safe-env:\n"
        f"- max_usd: {caps.get('max_usd')}\n"
        f"- max_model_tokens: {caps.get('max_model_tokens')}\n"
        f"- max_tool_calls: {caps.get('max_tool_calls')}\n"
        f"- max_wall_seconds: {caps.get('max_wall_seconds')}\n\n"
        "Perform the bounded spec-authoring work requested by the safe-env artifacts in this workspace. "
        "Keep all edits inside the allowed write roots."
    )


def _repair_prompt(
    *,
    component: str,
    pact_project: str,
    source_roots: Sequence[str],
    allowed_context: Mapping[str, object],
    forbidden_writes: Sequence[str],
    caps: Mapping[str, object],
) -> str:
    return (
        "You are the Pact BURN-E repair agent.\n\n"
        "Hard rules:\n"
        f"- Component: {component}\n"
        f"- Pact project, read-only for repair: {pact_project}\n"
        f"- Allowed implementation write roots: {', '.join(source_roots)}\n"
        f"- Forbidden write classes: {', '.join(forbidden_writes)}\n"
        "- Do not edit contracts, visible tests, workflow files, control-plane files, hidden-oracle files, or agent-safe files.\n"
        "- Do not run broad workspace scans. Inspect only the listed source roots, Pact project, and safe-env artifact inputs.\n"
        "- If the repair requires broader context, stop and explain that the allowed context is insufficient.\n\n"
        "Spend caps supplied by safe-env:\n"
        f"- max_usd: {caps.get('max_usd')}\n"
        f"- max_model_tokens: {caps.get('max_model_tokens')}\n"
        f"- max_tool_calls: {caps.get('max_tool_calls')}\n"
        f"- max_wall_seconds: {caps.get('max_wall_seconds')}\n\n"
        "Allowed context JSON:\n"
        f"{json.dumps(allowed_context, indent=2, sort_keys=True)}\n\n"
        "Perform the bounded repair requested by the safe-env artifacts in this workspace. "
        "Keep all edits inside the allowed implementation write roots."
    )


def _run_burne_agent(
    *,
    mode: str,
    args: Any,
    env: Mapping[str, str] | None = None,
) -> int:
    env = dict(os.environ if env is None else env)
    cwd = Path.cwd().resolve()
    violations: list[dict[str, str]] = []
    caps = parse_caps(env, violations)
    component = _validate_component(env, violations)
    pact_project = _validate_pact_project(env, cwd, component, violations)
    output_path = _validate_output_path(str(getattr(args, "output", "") or ""), cwd, violations)

    source_roots: list[ResolvedPath] = []
    allowed_context: dict[str, object] = {}
    forbidden_writes: list[str] = []
    if mode == "spec-author":
        if getattr(args, "source_root", None):
            violations.append(violation("source-root", "spec-author does not accept --source-root"))
        role = str(env.get("AGENT_SAFE_SPEC_AGENT_ROLE", "") or "").strip()
        if role and role != "spec-agent":
            violations.append(violation("AGENT_SAFE_SPEC_AGENT_ROLE", "spec author role must be spec-agent"))
    else:
        source_roots = _validate_source_roots(
            list(getattr(args, "source_root", []) or []),
            cwd,
            pact_project,
            violations,
        )
        allowed_context = _validate_repair_allowed_context(env, violations)
        forbidden_writes = _validate_repair_forbidden_writes(env, violations)

    if shutil.which("codex") is None:
        violations.append(violation("codex", "codex CLI not found on PATH"))

    report = _base_report(mode, cwd, caps, component, pact_project)
    report["source_roots"] = [root.relative for root in source_roots]
    allowed_roots = [pact_project.relative] if mode == "spec-author" and pact_project else [root.relative for root in source_roots]
    report["allowed_write_roots"] = allowed_roots

    if violations:
        report["status"] = "policy-failed"
        report["policy"] = {"passed": False, "violations": violations}
        _emit_report(report, output_path)
        return 2

    assert pact_project is not None
    before = workspace_manifest(cwd)
    if mode == "spec-author":
        prompt = _spec_prompt(
            component=component,
            pact_project=pact_project.relative,
            caps=caps,
            allowed_roots=allowed_roots,
        )
    else:
        prompt = _repair_prompt(
            component=component,
            pact_project=pact_project.relative,
            source_roots=[root.relative for root in source_roots],
            allowed_context=allowed_context,
            forbidden_writes=forbidden_writes,
            caps=caps,
        )

    command, result = run_codex_once(cwd, prompt, int(caps["max_wall_seconds"]))
    report["agent"] = {
        "provider": "codex",
        "command": command,
        "status": result.status,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": result.output,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
    }

    after = workspace_manifest(cwd)
    forbidden_changed_paths = changed_paths_outside_roots(before, after, allowed_roots)
    report["write_guard"] = {
        "status": "passed" if not forbidden_changed_paths else "failed",
        "forbidden_changed_paths": forbidden_changed_paths,
    }
    if forbidden_changed_paths:
        violations.append(violation("write-guard", "agent changed files outside allowed write roots"))

    accepted = result.status == "passed" and result.exit_code == 0 and not violations
    report["accepted"] = accepted
    report["status"] = "passed" if accepted else ("timeout" if result.status == "timeout" else "failed")
    report["policy"] = {"passed": not violations, "violations": violations}
    _emit_report(report, output_path)

    if accepted:
        return 0
    if result.status == "timeout":
        return 124
    if forbidden_changed_paths:
        return 3
    return result.exit_code or 1


def run_burne_spec_author(args: Any, env: Mapping[str, str] | None = None) -> int:
    return _run_burne_agent(mode="spec-author", args=args, env=env)


def run_burne_repair(args: Any, env: Mapping[str, str] | None = None) -> int:
    return _run_burne_agent(mode="repair", args=args, env=env)
