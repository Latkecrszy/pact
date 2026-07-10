from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pact import burne


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "pact" / "api" / "contracts").mkdir(parents=True)
    (tmp_path / "pact" / "api" / "tests").mkdir(parents=True)
    (tmp_path / "services" / "api").mkdir(parents=True)
    (tmp_path / "services" / "api" / "handler.py").write_text("def handler():\n    return True\n", encoding="utf-8")
    return tmp_path


def _env(**overrides: str) -> dict[str, str]:
    values = {
        "AGENT_SAFE_AGENT_MAX_WALL_SECONDS": "30",
        "AGENT_SAFE_AGENT_MAX_MODEL_TOKENS": "5000",
        "AGENT_SAFE_AGENT_MAX_TOOL_CALLS": "10",
        "AGENT_SAFE_AGENT_MAX_USD": "0.25",
        "AGENT_SAFE_COMPONENT": "api",
        "AGENT_SAFE_PACT_COMPONENT_ID": "api",
        "AGENT_SAFE_PACT_PROJECT": "pact/api",
        "AGENT_SAFE_SPEC_AGENT_ROLE": "spec-agent",
        "AGENT_SAFE_REPAIR_AGENT_ALLOWED_CONTEXT": json.dumps({"issue_context_ref": "triage.json"}),
        "AGENT_SAFE_REPAIR_AGENT_FORBIDDEN_WRITES": "contracts,visible-tests,control-plane,hidden-oracle",
    }
    values.update(overrides)
    return values


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {"output": "", "source_root": []}
    values.update(overrides)
    return argparse.Namespace(**values)


def _successful_codex(captured: dict[str, object]):
    def fake_run(command, *, input, capture_output, text, cwd, env, timeout, check):  # noqa: A002
        captured["command"] = command
        captured["input"] = input
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("agent completed", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="stdout", stderr="")

    return fake_run


def _load_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_exposes_burne_commands(capsys):
    from pact.cli import main

    with patch.object(sys, "argv", ["pact", "burne", "--help"]):
        with pytest.raises(SystemExit) as error:
            main()

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "spec-author" in output
    assert "repair" in output


def test_spec_author_fails_closed_without_required_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        result = burne.run_burne_spec_author(_args(), env={})

    assert result == 2


def test_caps_reject_values_above_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "report.json"

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        result = burne.run_burne_spec_author(_args(output="report.json"), env=_env(AGENT_SAFE_AGENT_MAX_USD="1.01"))

    report = _load_report(output)
    assert result == 2
    assert report["status"] == "policy-failed"
    assert any(item["name"] == "AGENT_SAFE_AGENT_MAX_USD" for item in report["policy"]["violations"])


def test_pact_project_rejects_workspace_root_and_non_component_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        root_result = burne.run_burne_spec_author(_args(), env=_env(AGENT_SAFE_PACT_PROJECT="."))
        broad_result = burne.run_burne_spec_author(_args(), env=_env(AGENT_SAFE_PACT_PROJECT="pact"))

    assert root_result == 2
    assert broad_result == 2


def test_spec_author_rejects_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        result = burne.run_burne_spec_author(_args(source_root=["services/api"]), env=_env())

    assert result == 2


def test_repair_requires_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        result = burne.run_burne_repair(_args(), env=_env())

    assert result == 2


def test_repair_rejects_forbidden_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    (tmp_path / ".github").mkdir()
    monkeypatch.chdir(tmp_path)

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        result = burne.run_burne_repair(_args(source_root=[".github"]), env=_env())

    assert result == 2


def test_repair_prompt_and_codex_invocation_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}
    output = tmp_path / "repair-report.json"

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        with patch("pact.burne.subprocess.run", side_effect=_successful_codex(captured)):
            result = burne.run_burne_repair(
                _args(source_root=["services/api"], output="repair-report.json"),
                env=_env(),
            )

    report = _load_report(output)
    command = captured["command"]
    assert result == 0
    assert report["accepted"] is True
    assert command[:2] == ["codex", "exec"]
    assert command[command.index("-C") + 1] == str(tmp_path)
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert captured["timeout"] == 30
    assert "Allowed implementation write roots: services/api" in str(captured["input"])
    assert "max_usd: 0.25" in str(captured["input"])


def test_codex_timeout_returns_failed_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "timeout-report.json"

    def timeout_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"], output="partial", stderr="late")

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        with patch("pact.burne.subprocess.run", side_effect=timeout_run):
            result = burne.run_burne_repair(
                _args(source_root=["services/api"], output="timeout-report.json"),
                env=_env(),
            )

    report = _load_report(output)
    assert result == 124
    assert report["accepted"] is False
    assert report["status"] == "timeout"
    assert report["agent"]["status"] == "timeout"


def test_failed_codex_run_returns_failed_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "failed-report.json"

    def failed_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="failed")

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        with patch("pact.burne.subprocess.run", side_effect=failed_run):
            result = burne.run_burne_repair(
                _args(source_root=["services/api"], output="failed-report.json"),
                env=_env(),
            )

    report = _load_report(output)
    assert result == 7
    assert report["accepted"] is False
    assert report["status"] == "failed"
    assert report["agent"]["exit_code"] == 7


def test_write_guard_rejects_forbidden_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "write-guard-report.json"

    def bad_run(command, **kwargs):
        (tmp_path / ".github" / "workflows" / "owned.yml").write_text("name: bad\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with patch("pact.burne.shutil.which", return_value="/usr/bin/codex"):
        with patch("pact.burne.subprocess.run", side_effect=bad_run):
            result = burne.run_burne_repair(
                _args(source_root=["services/api"], output="write-guard-report.json"),
                env=_env(),
            )

    report = _load_report(output)
    assert result == 3
    assert report["accepted"] is False
    assert report["write_guard"]["status"] == "failed"
    assert ".github/workflows/owned.yml" in report["write_guard"]["forbidden_changed_paths"]
