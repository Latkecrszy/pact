"""Codex CLI backend for agentic code implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from pact.backends.agent_runtime import (
    AgentRuntimeSpec,
    build_codex_exec_command,
    normalize_model_for_backend,
    require_agent_cli,
    wrap_worker_prompt,
)
from pact.backends.openai import (
    _is_strict_compatible,
    _prepare_strict_schema,
    _strip_nulls,
)
from pact.budget import BudgetExceeded, BudgetTracker

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


def prepare_codex_output_schema(schema: type[BaseModel]) -> dict | None:
    """Build a Codex-compatible strict schema, or None for JSON-mode fallback."""
    output_schema = schema.model_json_schema()
    if not _is_strict_compatible(output_schema):
        return None
    _prepare_strict_schema(output_schema)
    return output_schema


class CodexCodeBackend:
    """Backend using `codex exec` with repository tool access."""

    def __init__(
        self,
        budget: BudgetTracker,
        model: str = "",
        repo_path: Path | None = None,
        timeout: int = 600,
        max_retries: int = 2,
        session_id: str = "",
    ) -> None:
        self._model = normalize_model_for_backend("codex_code", model)
        self._budget = budget
        self._repo_path = Path(repo_path) if repo_path else None
        self._timeout = timeout
        self._max_retries = max_retries
        self._session_id = session_id

    def set_model(self, model: str) -> None:
        self._model = normalize_model_for_backend("codex_code", model)

    def set_repo_path(self, path: Path) -> None:
        self._repo_path = path

    async def assess(
        self,
        schema: type[T],
        prompt: str,
        system: str,
        max_tokens: int = 32768,
    ) -> tuple[T, int, int]:
        """Call Codex with schema enforcement via `--output-schema`."""
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await self._assess_once(schema, prompt, system, max_tokens)
            except RuntimeError as exc:
                last_err = exc
                if "timed out" in str(exc) and attempt < self._max_retries - 1:
                    logger.warning(
                        "Codex assess attempt %d/%d timed out, retrying...",
                        attempt + 1,
                        self._max_retries,
                    )
                    continue
                raise
        raise last_err  # type: ignore[misc]

    async def _assess_once(
        self,
        schema: type[T],
        prompt: str,
        system: str,
        max_tokens: int = 32768,
    ) -> tuple[T, int, int]:
        del max_tokens  # Codex CLI owns output limits.
        require_agent_cli("codex")

        output_schema = prepare_codex_output_schema(schema)
        full_prompt = (
            f"{system}\n\n"
            f"Task:\n{prompt}\n\n"
            "Respond only with the JSON object matching the provided schema."
        )
        if output_schema is None:
            full_prompt += (
                "\n\nJSON schema:\n"
                f"{json.dumps(schema.model_json_schema(), indent=2)}"
            )

        cwd = self._repo_path or Path.cwd()
        with tempfile.TemporaryDirectory(prefix="pact-codex-") as tmp:
            schema_path = Path(tmp) / "schema.json"
            output_path = Path(tmp) / "last_message.json"
            if output_schema is not None:
                schema_path.write_text(json.dumps(output_schema))

            spec = AgentRuntimeSpec(
                provider="codex",
                model=self._model,
                working_dir=cwd,
                session_id=self._session_id,
                sandbox="read-only",
                output_schema=schema_path if output_schema is not None else None,
                output_file=output_path,
            )
            cmd = build_codex_exec_command(spec)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=spec.env(),
            )

            raw_prompt = wrap_worker_prompt(full_prompt, self._session_id)
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=raw_prompt.encode()),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError(f"codex exec timed out after {self._timeout}s")

            if proc.returncode != 0:
                error = stderr.decode(errors="replace")
                raise RuntimeError(
                    f"codex exec failed (exit {proc.returncode}): "
                    f"{error[-2000:]}"
                )

            raw = output_path.read_text() if output_path.exists() else stdout.decode()
            data = self._extract_json(raw)
            _strip_nulls(data)
            in_tok = 0
            out_tok = 0
            if not self._budget.record_tokens_validated(
                in_tok,
                out_tok,
                prompt_text=raw_prompt,
                response_text=raw,
            ):
                raise BudgetExceeded("Budget exceeded after Codex CLI call")
            return schema.model_validate(data), in_tok, out_tok

    @staticmethod
    def _extract_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"```(?:json)?\s*\n({.*?})\s*\n```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break
        raise RuntimeError(f"No valid JSON found: {text[:200]}")

    async def implement(
        self,
        prompt: str,
        working_dir: Path | None = None,
        max_turns: int = 30,
        timeout: int = 600,
    ) -> tuple[str, int, int]:
        """Run an iterative Codex session with workspace write access."""
        require_agent_cli("codex")
        cwd = Path(working_dir or self._repo_path or Path.cwd())
        with tempfile.TemporaryDirectory(prefix="pact-codex-") as tmp:
            output_path = Path(tmp) / "last_message.txt"
            spec = AgentRuntimeSpec(
                provider="codex",
                model=self._model,
                working_dir=cwd,
                session_id=self._session_id,
                max_turns=max_turns,
                sandbox="workspace-write",
                output_file=output_path,
            )
            cmd = build_codex_exec_command(spec)
            wrapped_prompt = wrap_worker_prompt(prompt, self._session_id)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=spec.env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=wrapped_prompt.encode()),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "codex implement timed out after %ds (pid=%s), killing",
                    timeout,
                    proc.pid,
                )
                proc.kill()
                await proc.wait()
                raise RuntimeError(f"codex implement timed out after {timeout}s")

            if proc.returncode != 0:
                error = stderr.decode(errors="replace")
                raise RuntimeError(
                    f"codex implement failed (exit {proc.returncode}): "
                    f"{error[-2000:]}"
                )

            raw = output_path.read_text() if output_path.exists() else stdout.decode()
            in_tok = 0
            out_tok = 0
            if not self._budget.record_tokens_validated(
                in_tok,
                out_tok,
                prompt_text=wrapped_prompt,
                response_text=raw,
            ):
                raise BudgetExceeded("Budget exceeded after Codex CLI call")
            return raw, in_tok, out_tok

    async def close(self) -> None:
        pass
