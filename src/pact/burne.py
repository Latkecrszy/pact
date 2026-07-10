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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from pact.budget import get_model_pricing_table

MAX_WALL_SECONDS = 900
MAX_MODEL_TOKENS = 50_000
MAX_TOOL_CALLS = 75
MAX_USD_CENTS = 100
REPORT_SCHEMA_VERSION = "pact-burne-agent/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
MAX_CONTEXT_FILES = 80
MAX_CONTEXT_BYTES = 180_000
MAX_FILE_BYTES = 60_000
MIN_OUTPUT_TOKENS = 16
SKIPPED_CONTEXT_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

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
    output: str
    elapsed_seconds: float
    response_id: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    applied_paths: tuple[str, ...]


@dataclass(frozen=True)
class ContextFile:
    path: str
    content: str
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class OpenAIRequestPlan:
    model: str
    input_token_bound: int
    max_output_tokens: int
    estimated_max_cost_usd: float


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


def _path_allowed(relative: str, allowed_roots: Sequence[str]) -> bool:
    return any(relative == root or relative.startswith(f"{root}/") for root in allowed_roots)


def _resolved_allowed_roots(cwd: Path, allowed_roots: Sequence[str]) -> list[Path]:
    workspace_root = cwd.resolve()
    return [(workspace_root / root).resolve() for root in allowed_roots]


def _path_has_existing_symlink_component(cwd: Path, relative: str) -> bool:
    current = cwd.resolve()
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _resolve_openai_model(env: Mapping[str, str]) -> str:
    return str(
        env.get("AGENT_SAFE_OPENAI_MODEL")
        or env.get("AGENT_SAFE_AGENT_MODEL")
        or DEFAULT_OPENAI_MODEL
    ).strip()


def _pricing_for_model_strict(model: str) -> tuple[float, float] | None:
    table = get_model_pricing_table()
    for key in sorted(table, key=len, reverse=True):
        if model == key or model.startswith(f"{key}-"):
            return table[key]
    return None


def _bounded_token_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    pricing = _pricing_for_model_strict(model)
    if pricing is None:
        return float("inf")
    input_cost, output_cost = pricing
    return (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000


def _plan_openai_request(
    *,
    env: Mapping[str, str],
    caps: Mapping[str, object],
    input_text: str,
    violations: list[dict[str, str]],
) -> OpenAIRequestPlan | None:
    model = _resolve_openai_model(env)
    if not model:
        violations.append(violation("AGENT_SAFE_OPENAI_MODEL", "OpenAI model is required"))
        return None
    pricing = _pricing_for_model_strict(model)
    if pricing is None:
        violations.append(
            violation(
                "AGENT_SAFE_OPENAI_MODEL",
                f"OpenAI model {model!r} has no configured pricing; refusing to run spend-capped agent",
            )
        )
        return None

    input_token_bound = len(input_text.encode("utf-8"))
    max_model_tokens = int(caps.get("max_model_tokens") or 0)
    max_usd_cents = int(caps.get("max_usd_cents") or 0)
    if input_token_bound >= max_model_tokens:
        violations.append(
            violation(
                "AGENT_SAFE_AGENT_MAX_MODEL_TOKENS",
                "bounded input context exhausts AGENT_SAFE_AGENT_MAX_MODEL_TOKENS",
            )
        )
        return None

    input_cost, output_cost = pricing
    if output_cost <= 0:
        violations.append(violation("AGENT_SAFE_OPENAI_MODEL", "OpenAI model output pricing must be > 0"))
        return None

    max_by_tokens = max_model_tokens - input_token_bound
    max_usd = max_usd_cents / 100
    input_cost_bound = input_token_bound * input_cost / 1_000_000
    remaining_usd = max_usd - input_cost_bound
    max_by_cost = int((remaining_usd * 1_000_000) / output_cost)
    max_output_tokens = min(max_by_tokens, max_by_cost)
    if max_output_tokens < MIN_OUTPUT_TOKENS:
        violations.append(
            violation(
                "AGENT_SAFE_AGENT_MAX_USD",
                "bounded input context plus minimum output cannot fit inside AGENT_SAFE_AGENT_MAX_USD",
            )
        )
        return None

    return OpenAIRequestPlan(
        model=model,
        input_token_bound=input_token_bound,
        max_output_tokens=max_output_tokens,
        estimated_max_cost_usd=_bounded_token_cost(
            model=model,
            input_tokens=input_token_bound,
            output_tokens=max_output_tokens,
        ),
    )


def _read_context_file(path: Path) -> tuple[str, str, int] | None:
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{path} exceeds MAX_FILE_BYTES")
    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return hashlib.sha256(data).hexdigest(), text, len(data)


def _iter_context_file_paths(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if name not in SKIPPED_CONTEXT_DIRS
            and not (Path(dirpath) / name).is_symlink()
        ]
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def _collect_context_files(
    cwd: Path,
    roots: Sequence[ResolvedPath],
    violations: list[dict[str, str]],
) -> list[ContextFile]:
    files: list[ContextFile] = []
    total_bytes = 0
    raw_candidates = 0
    workspace_root = cwd.resolve()
    for root in roots:
        for path in _iter_context_file_paths(root.resolved):
            raw_candidates += 1
            if raw_candidates > MAX_CONTEXT_FILES:
                violations.append(violation("context", f"context candidate files exceed MAX_CONTEXT_FILES={MAX_CONTEXT_FILES}"))
                return files
            relative_parts = path.relative_to(root.resolved).parts
            if any(part in SKIPPED_CONTEXT_DIRS for part in relative_parts):
                continue
            if not path.is_file() or path.is_symlink():
                continue
            resolved = path.resolve()
            if not _path_is_relative_to(resolved, workspace_root):
                violations.append(violation("context", f"context file {path} escapes workspace"))
                continue
            try:
                loaded = _read_context_file(path)
            except ValueError:
                violations.append(violation("context", f"context file {path.relative_to(workspace_root).as_posix()} is too large"))
                continue
            except OSError as error:
                violations.append(violation("context", f"context file {path.relative_to(workspace_root).as_posix()} cannot be read: {error}"))
                continue
            if loaded is None:
                continue
            digest, content, byte_count = loaded
            total_bytes += byte_count
            if total_bytes > MAX_CONTEXT_BYTES:
                violations.append(violation("context", f"context exceeds MAX_CONTEXT_BYTES={MAX_CONTEXT_BYTES}"))
                return files
            if len(files) >= MAX_CONTEXT_FILES:
                violations.append(violation("context", f"context exceeds MAX_CONTEXT_FILES={MAX_CONTEXT_FILES}"))
                return files
            files.append(
                ContextFile(
                    path=path.relative_to(workspace_root).as_posix(),
                    content=content,
                    sha256=digest,
                    byte_count=byte_count,
                )
            )
    return sorted(files, key=lambda item: item.path)


def _response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "summary", "changes"],
        "properties": {
            "status": {"type": "string", "enum": ["changed", "no_change", "blocked"]},
            "summary": {"type": "string"},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
    }


def _build_agent_payload(
    *,
    mode: str,
    component: str,
    pact_project: str,
    allowed_roots: Sequence[str],
    source_roots: Sequence[str],
    allowed_context: Mapping[str, object],
    forbidden_writes: Sequence[str],
    caps: Mapping[str, object],
    context_files: Sequence[ContextFile],
) -> str:
    payload = {
        "mode": mode,
        "component": component,
        "pact_project": pact_project,
        "allowed_write_roots": list(allowed_roots),
        "source_roots": list(source_roots),
        "allowed_context": dict(allowed_context),
        "forbidden_writes": list(forbidden_writes),
        "caps": {
            "max_usd": caps.get("max_usd"),
            "max_model_tokens": caps.get("max_model_tokens"),
            "max_tool_calls": caps.get("max_tool_calls"),
            "max_wall_seconds": caps.get("max_wall_seconds"),
        },
        "files": [
            {
                "path": item.path,
                "sha256": item.sha256,
                "byte_count": item.byte_count,
                "content": item.content,
            }
            for item in context_files
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _system_instructions(mode: str) -> str:
    if mode == "spec-author":
        return (
            "You are the Pact BURN-E spec-author agent. You receive only a bounded "
            "file snapshot. Return a JSON object that either blocks with a short "
            "reason or provides full UTF-8 file contents for files to write. Do not "
            "request or assume broader repository context. Do not modify source code."
        )
    return (
        "You are the Pact BURN-E repair agent. You receive only a bounded file "
        "snapshot and allowed context. Return a JSON object that either blocks with "
        "a short reason or provides full UTF-8 file contents for implementation files "
        "to write. Do not edit contracts, tests, workflows, control-plane files, "
        "hidden-oracle files, or agent-safe files."
    )


def _post_openai_response(
    request_payload: Mapping[str, object],
    *,
    api_key: str,
    timeout_seconds: int,
) -> dict[str, object]:
    data = json.dumps(request_payload).encode("utf-8")
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_response_text(payload: Mapping[str, object]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text
    parts: list[str] = []
    for item in payload.get("output", []) if isinstance(payload.get("output"), list) else []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _usage_tokens(payload: Mapping[str, object]) -> tuple[int, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return int(input_tokens or 0), int(output_tokens or 0)


def _parse_agent_json(text: str) -> dict[str, object]:
    return json.loads(text)


def _apply_agent_changes(
    *,
    cwd: Path,
    payload: Mapping[str, object],
    allowed_roots: Sequence[str],
    violations: list[dict[str, str]],
) -> tuple[str, tuple[str, ...]]:
    status = payload.get("status")
    summary = str(payload.get("summary", "") or "")
    if status == "blocked":
        violations.append(violation("agent", summary or "agent blocked due to insufficient context"))
        return summary, ()
    if status == "no_change":
        return summary, ()
    if status != "changed":
        violations.append(violation("agent", "agent returned invalid status"))
        return summary, ()

    changes = payload.get("changes")
    if not isinstance(changes, list):
        violations.append(violation("agent", "agent changes must be a list"))
        return summary, ()

    prepared: list[tuple[Path, str, str]] = []
    resolved_allowed_roots = _resolved_allowed_roots(cwd, allowed_roots)
    for change in changes:
        if not isinstance(change, dict):
            violations.append(violation("agent", "agent change must be an object"))
            continue
        path_value = str(change.get("path", "") or "")
        content = change.get("content")
        path = _clean_relative_path(
            path_value,
            field_name="agent-change-path",
            cwd=cwd,
            violations=violations,
            require_exists=False,
            require_dir=False,
        )
        if path is None:
            continue
        if not _path_allowed(path.relative, allowed_roots):
            violations.append(violation("write-guard", f"agent attempted forbidden write: {path.relative}"))
            continue
        if _path_has_existing_symlink_component(cwd, path.relative):
            violations.append(violation("write-guard", f"agent attempted symlinked write path: {path.relative}"))
            continue
        if not any(_path_is_relative_to(path.resolved, root) for root in resolved_allowed_roots):
            violations.append(violation("write-guard", f"agent attempted resolved write outside allowed roots: {path.relative}"))
            continue
        if path.resolved.exists() and path.resolved.is_dir():
            violations.append(violation("write-guard", f"agent attempted to write directory: {path.relative}"))
            continue
        if not isinstance(content, str):
            violations.append(violation("agent", f"agent content for {path.relative} must be a string"))
            continue
        if "\x00" in content:
            violations.append(violation("agent", f"agent content for {path.relative} contains NUL"))
            continue
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            violations.append(violation("agent", f"agent content for {path.relative} exceeds MAX_FILE_BYTES"))
            continue
        prepared.append((path.resolved, path.relative, content))

    if violations:
        return summary, ()

    applied: list[str] = []
    for resolved, relative, content in prepared:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        applied.append(relative)
    return summary, tuple(applied)


def run_openai_agent_once(
    *,
    mode: str,
    cwd: Path,
    component: str,
    pact_project: str,
    allowed_roots: Sequence[str],
    source_roots: Sequence[str],
    allowed_context: Mapping[str, object],
    forbidden_writes: Sequence[str],
    caps: Mapping[str, object],
    context_files: Sequence[ContextFile],
    env: Mapping[str, str],
    timeout_seconds: int,
    violations: list[dict[str, str]],
) -> tuple[OpenAIRequestPlan | None, AgentResult | None]:
    agent_payload = _build_agent_payload(
        mode=mode,
        component=component,
        pact_project=pact_project,
        allowed_roots=allowed_roots,
        source_roots=source_roots,
        allowed_context=allowed_context,
        forbidden_writes=forbidden_writes,
        caps=caps,
        context_files=context_files,
    )
    instructions = _system_instructions(mode)
    response_schema = _response_schema()
    planning_input = json.dumps(
        {
            "instructions": instructions,
            "input": agent_payload,
            "text_format": response_schema,
        },
        sort_keys=True,
    )
    request_plan = _plan_openai_request(
        env=env,
        caps=caps,
        input_text=planning_input,
        violations=violations,
    )
    if request_plan is None:
        return None, None

    api_key = str(env.get("OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        violations.append(violation("OPENAI_API_KEY", "OPENAI_API_KEY is required for BURN-E OpenAI API agents"))
        return request_plan, None

    request_payload: dict[str, object] = {
        "model": request_plan.model,
        "instructions": instructions,
        "input": agent_payload,
        "max_output_tokens": request_plan.max_output_tokens,
        "max_tool_calls": int(caps.get("max_tool_calls") or 0),
        "store": False,
        "truncation": "disabled",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "burne_file_changes",
                "strict": True,
                "schema": response_schema,
            },
        },
    }

    started = time.monotonic()
    try:
        response_payload = _post_openai_response(
            request_payload,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    except TimeoutError:
        elapsed = time.monotonic() - started
        return request_plan, AgentResult(
            status="timeout",
            exit_code=124,
            output="",
            elapsed_seconds=elapsed,
            response_id="",
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            applied_paths=(),
        )
    except urllib.error.URLError as error:
        elapsed = time.monotonic() - started
        return request_plan, AgentResult(
            status="failed",
            exit_code=1,
            output=_short_text(error),
            elapsed_seconds=elapsed,
            response_id="",
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            applied_paths=(),
        )

    elapsed = time.monotonic() - started
    response_text = _extract_response_text(response_payload)
    input_tokens, output_tokens = _usage_tokens(response_payload)
    estimated_cost = _bounded_token_cost(
        model=request_plan.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    if input_tokens + output_tokens > int(caps.get("max_model_tokens") or 0):
        violations.append(violation("budget", "OpenAI response exceeded AGENT_SAFE_AGENT_MAX_MODEL_TOKENS"))
    if estimated_cost > int(caps.get("max_usd_cents") or 0) / 100:
        violations.append(violation("budget", "OpenAI response exceeded AGENT_SAFE_AGENT_MAX_USD"))

    try:
        agent_json = _parse_agent_json(response_text)
    except json.JSONDecodeError as error:
        violations.append(violation("agent", f"agent returned invalid JSON: {error}"))
        return request_plan, AgentResult(
            status="failed",
            exit_code=1,
            output=_short_text(response_text),
            elapsed_seconds=elapsed,
            response_id=str(response_payload.get("id", "") or ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            applied_paths=(),
        )

    summary, applied_paths = _apply_agent_changes(
        cwd=cwd,
        payload=agent_json,
        allowed_roots=allowed_roots,
        violations=violations,
    )
    status = "passed" if not violations else "failed"
    return request_plan, AgentResult(
        status=status,
        exit_code=0 if status == "passed" else 1,
        output=_short_text(summary or response_text),
        elapsed_seconds=elapsed,
        response_id=str(response_payload.get("id", "") or ""),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
        applied_paths=applied_paths,
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
        "context": {
            "files": 0,
            "bytes": 0,
            "max_files": MAX_CONTEXT_FILES,
            "max_bytes": MAX_CONTEXT_BYTES,
        },
        "agent": {
            "provider": "openai-responses-api",
            "model": "",
            "request": {
                "endpoint": "POST /v1/responses",
                "max_output_tokens": 0,
                "max_tool_calls": 0,
                "input_token_bound": 0,
                "estimated_max_cost_usd": 0.0,
                "truncation": "disabled",
            },
            "status": "not-run",
            "exit_code": None,
            "output": "",
            "elapsed_seconds": 0.0,
            "response_id": "",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
            "applied_paths": [],
        },
    }


def _emit_report(report: Mapping[str, object], output_path: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path is None:
        print(text, end="")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


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

    report = _base_report(mode, cwd, caps, component, pact_project)
    report["source_roots"] = [root.relative for root in source_roots]
    allowed_roots = [pact_project.relative] if mode == "spec-author" and pact_project else [root.relative for root in source_roots]
    report["allowed_write_roots"] = allowed_roots

    context_roots = [pact_project] if mode == "spec-author" and pact_project else [root for root in [pact_project, *source_roots] if root]
    context_files = _collect_context_files(cwd, context_roots, violations)
    report["context"] = {
        "files": len(context_files),
        "bytes": sum(item.byte_count for item in context_files),
        "max_files": MAX_CONTEXT_FILES,
        "max_bytes": MAX_CONTEXT_BYTES,
    }

    if violations:
        report["status"] = "policy-failed"
        report["policy"] = {"passed": False, "violations": violations}
        _emit_report(report, output_path)
        return 2

    assert pact_project is not None
    request_plan, result = run_openai_agent_once(
        mode=mode,
        cwd=cwd,
        component=component,
        pact_project=pact_project.relative,
        allowed_roots=allowed_roots,
        source_roots=[root.relative for root in source_roots],
        allowed_context=allowed_context,
        forbidden_writes=forbidden_writes,
        caps=caps,
        context_files=context_files,
        env=env,
        timeout_seconds=int(caps["max_wall_seconds"]),
        violations=violations,
    )
    if request_plan is not None:
        report["agent"]["model"] = request_plan.model
        report["agent"]["request"] = {
            "endpoint": "POST /v1/responses",
            "max_output_tokens": request_plan.max_output_tokens,
            "max_tool_calls": int(caps.get("max_tool_calls") or 0),
            "input_token_bound": request_plan.input_token_bound,
            "estimated_max_cost_usd": round(request_plan.estimated_max_cost_usd, 6),
            "truncation": "disabled",
        }
    if result is None:
        report["status"] = "policy-failed"
        report["policy"] = {"passed": False, "violations": violations}
        _emit_report(report, output_path)
        return 2
    report["agent"] |= {
        "status": result.status,
        "exit_code": result.exit_code,
        "output": result.output,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "response_id": result.response_id,
        "usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "estimated_cost_usd": round(result.estimated_cost_usd, 6),
        },
        "applied_paths": list(result.applied_paths),
    }

    write_guard_violations = [item for item in violations if item.get("name") == "write-guard"]
    attempted_forbidden_paths = [
        item["message"].removeprefix("agent attempted forbidden write: ")
        for item in write_guard_violations
        if isinstance(item.get("message"), str)
        and item.get("message", "").startswith("agent attempted forbidden write: ")
    ]
    report["write_guard"] = {
        "status": "passed" if not write_guard_violations else "failed",
        "forbidden_changed_paths": sorted(set(attempted_forbidden_paths)),
    }

    accepted = result.status == "passed" and result.exit_code == 0 and not violations
    report["accepted"] = accepted
    report["status"] = "passed" if accepted else ("timeout" if result.status == "timeout" else "failed")
    report["policy"] = {"passed": not violations, "violations": violations}
    _emit_report(report, output_path)

    if accepted:
        return 0
    if result.status == "timeout":
        return 124
    if write_guard_violations:
        return 3
    return result.exit_code or 1


def run_burne_spec_author(args: Any, env: Mapping[str, str] | None = None) -> int:
    return _run_burne_agent(mode="spec-author", args=args, env=env)


def run_burne_repair(args: Any, env: Mapping[str, str] | None = None) -> int:
    return _run_burne_agent(mode="repair", args=args, env=env)
