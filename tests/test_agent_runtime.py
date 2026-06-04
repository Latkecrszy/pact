"""Tests for provider-neutral coding-agent runtimes."""

from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from pact.backends.agent_runtime import (
    AgentRuntimeSpec,
    build_claude_implement_command,
    build_codex_exec_command,
    build_tmux_agent_command,
    build_tmux_launch_command,
    collaboration_preamble,
    is_iterative_backend,
    normalize_model_for_backend,
    provider_for_backend,
    require_agent_cli,
    session_id_for_project,
)
from pact.backends.codex_code import prepare_codex_output_schema


class StrictSchema(BaseModel):
    ok: bool


class FreeFormSchema(BaseModel):
    values: dict[str, str]


class TestRuntimeCapabilities:
    def test_iterative_backends_include_claude_and_codex_shells(self):
        assert is_iterative_backend("claude_code")
        assert is_iterative_backend("claude_code_team")
        assert is_iterative_backend("codex_code")
        assert is_iterative_backend("codex_code_team")
        assert not is_iterative_backend("openai")

    def test_provider_mapping(self):
        assert provider_for_backend("claude_code") == "claude"
        assert provider_for_backend("codex_code") == "codex"
        assert provider_for_backend("openai") == "openai"

    def test_codex_inherits_default_for_claude_model(self):
        assert normalize_model_for_backend("codex_code", "claude-opus-4-6") == ""
        assert normalize_model_for_backend("codex_code", "gpt-5.2") == "gpt-5.2"

    def test_codex_output_schema_is_strict(self):
        schema = prepare_codex_output_schema(StrictSchema)
        assert schema is not None
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["ok"]

    def test_codex_free_form_schema_uses_json_mode(self):
        assert prepare_codex_output_schema(FreeFormSchema) is None

    def test_missing_agent_cli_fails_with_setup_guidance(self):
        with patch("pact.backends.agent_runtime.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="codex CLI not found on PATH"):
                require_agent_cli("codex")

    def test_fallback_session_id_is_stable_and_path_scoped(self, tmp_path: Path):
        first = SimpleNamespace(project_dir=tmp_path / "one" / "same-name")
        second = SimpleNamespace(project_dir=tmp_path / "two" / "same-name")

        assert session_id_for_project(first) == session_id_for_project(first)
        assert session_id_for_project(first) != session_id_for_project(second)


class TestRuntimeSpec:
    def test_env_carries_shared_session(self, tmp_path: Path):
        spec = AgentRuntimeSpec(
            provider="codex",
            model="gpt-5.2",
            working_dir=tmp_path,
            session_id="pact-test-session",
        )
        env = spec.env()
        assert env["SIGNET_SESSION"] == "pact-test-session"
        assert env["PACT_SESSION_ID"] == "pact-test-session"
        assert env["PACT_AGENT_PROVIDER"] == "codex"

    def test_collaboration_preamble_mentions_kindex_and_permission_tool(self):
        preamble = collaboration_preamble("pact-test-session")
        assert "Kindex" in preamble
        assert "permission tool" in preamble
        assert "pact-test-session" in preamble


class TestCommandBuilders:
    def test_claude_implement_command_preserves_existing_shape(self, tmp_path: Path):
        spec = AgentRuntimeSpec(
            provider="claude",
            model="claude-opus-4-6",
            working_dir=tmp_path,
            max_turns=17,
        )
        cmd = build_claude_implement_command(spec)
        assert cmd[:2] == ["claude", "-p"]
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--max-turns" in cmd
        assert "17" in cmd
        assert "Read,Write,Edit,Bash,Glob,Grep" in cmd

    def test_codex_exec_command_uses_workspace_and_output_file(self, tmp_path: Path):
        output = tmp_path / "last.txt"
        spec = AgentRuntimeSpec(
            provider="codex",
            model="gpt-5.2",
            working_dir=tmp_path,
            output_file=output,
        )
        cmd = build_codex_exec_command(spec)
        assert cmd[:2] == ["codex", "exec"]
        assert cmd[cmd.index("-C") + 1] == str(tmp_path)
        assert cmd[cmd.index("--model") + 1] == "gpt-5.2"
        assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"
        assert cmd[cmd.index("--output-last-message") + 1] == str(output)
        assert cmd[-1] == "-"

    def test_codex_exec_command_can_inherit_configured_model(self, tmp_path: Path):
        spec = AgentRuntimeSpec(
            provider="codex",
            model="",
            working_dir=tmp_path,
        )
        cmd = build_codex_exec_command(spec)
        assert "--model" not in cmd
        assert "--ask-for-approval" not in cmd

    def test_tmux_codex_command_reads_prompt_from_file(self, tmp_path: Path):
        prompt = tmp_path / "prompt.md"
        output = tmp_path / "out.txt"
        cmd = build_tmux_agent_command(
            provider="codex",
            prompt_file=prompt,
            output_file=str(output),
            model="gpt-5.2",
        )
        assert "codex exec" in cmd
        assert "--ask-for-approval" not in cmd
        assert f"< {prompt}" in cmd
        assert str(output) in cmd

    def test_tmux_codex_command_ignores_claude_default_model(self, tmp_path: Path):
        cmd = build_tmux_agent_command(
            provider="codex",
            prompt_file=tmp_path / "prompt.md",
            output_file=str(tmp_path / "out.txt"),
            model="claude-opus-4-6",
        )
        assert "--model" not in cmd

    def test_tmux_launch_quotes_working_dir_and_session(self):
        working_dir = "/tmp/project; touch /tmp/not-run"
        session_id = "session'; touch /tmp/not-run; '"
        cmd = build_tmux_launch_command("codex exec", working_dir, session_id)

        assert cmd.startswith(f"cd {shlex.quote(working_dir)} && ")
        assert f"SIGNET_SESSION={shlex.quote(session_id)}" in cmd
        assert f"PACT_SESSION_ID={shlex.quote(session_id)}" in cmd
