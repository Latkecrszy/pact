from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from pact import burne


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "pact" / "api" / "contracts").mkdir(parents=True)
    (tmp_path / "pact" / "api" / "tests").mkdir(parents=True)
    (tmp_path / "pact" / "api" / "contracts" / "contract.json").write_text(
        '{"component":"api"}\n',
        encoding="utf-8",
    )
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
        "OPENAI_API_KEY": "sk-test",
    }
    values.update(overrides)
    return values


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {"output": "", "source_root": []}
    values.update(overrides)
    return argparse.Namespace(**values)


def _response(agent_payload: dict[str, object], *, input_tokens: int = 100, output_tokens: int = 50) -> dict[str, object]:
    return {
        "id": "resp_test",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(agent_payload),
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def _successful_openai(captured: dict[str, object], agent_payload: dict[str, object]):
    def fake_post(request_payload, *, api_key, timeout_seconds):
        captured["request"] = request_payload
        captured["api_key"] = api_key
        captured["timeout_seconds"] = timeout_seconds
        return _response(agent_payload)

    return fake_post


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

    result = burne.run_burne_spec_author(_args(), env={})

    assert result == 2


def test_caps_reject_values_above_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "report.json"

    result = burne.run_burne_spec_author(_args(output="report.json"), env=_env(AGENT_SAFE_AGENT_MAX_USD="1.01"))

    report = _load_report(output)
    assert result == 2
    assert report["status"] == "policy-failed"
    assert any(item["name"] == "AGENT_SAFE_AGENT_MAX_USD" for item in report["policy"]["violations"])


def test_pact_project_rejects_workspace_root_and_non_component_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    root_result = burne.run_burne_spec_author(_args(), env=_env(AGENT_SAFE_PACT_PROJECT="."))
    broad_result = burne.run_burne_spec_author(_args(), env=_env(AGENT_SAFE_PACT_PROJECT="pact"))

    assert root_result == 2
    assert broad_result == 2


def test_spec_author_rejects_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = burne.run_burne_spec_author(_args(source_root=["services/api"]), env=_env())

    assert result == 2


def test_repair_requires_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = burne.run_burne_repair(_args(), env=_env())

    assert result == 2


def test_repair_rejects_forbidden_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    (tmp_path / ".github").mkdir()
    monkeypatch.chdir(tmp_path)

    result = burne.run_burne_repair(_args(source_root=[".github"]), env=_env())

    assert result == 2


def test_repair_openai_request_is_bounded_and_applies_allowed_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}
    output = tmp_path / "repair-report.json"
    new_handler = "def handler():\n    return 'fixed'\n"

    with patch(
        "pact.burne._post_openai_response",
        side_effect=_successful_openai(
            captured,
            {
                "status": "changed",
                "summary": "updated handler",
                "changes": [{"path": "services/api/handler.py", "content": new_handler}],
            },
        ),
    ):
        result = burne.run_burne_repair(
            _args(source_root=["services/api"], output="repair-report.json"),
            env=_env(),
        )

    report = _load_report(output)
    request = captured["request"]
    input_payload = json.loads(str(request["input"]))
    included_paths = {item["path"] for item in input_payload["files"]}

    assert result == 0
    assert report["accepted"] is True
    assert report["agent"]["provider"] == "openai-responses-api"
    assert report["agent"]["model"] == "gpt-4o-mini"
    assert report["agent"]["request"]["max_output_tokens"] == request["max_output_tokens"]
    assert request["max_output_tokens"] <= 5000
    assert request["max_tool_calls"] == 10
    assert request["store"] is False
    assert request["truncation"] == "disabled"
    assert captured["api_key"] == "sk-test"
    assert captured["timeout_seconds"] == 30
    assert "services/api/handler.py" in included_paths
    assert "pact/api/contracts/contract.json" in included_paths
    assert (tmp_path / "services" / "api" / "handler.py").read_text(encoding="utf-8") == new_handler
    assert report["agent"]["applied_paths"] == ["services/api/handler.py"]


def test_missing_openai_key_fails_before_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "missing-key-report.json"

    with patch("pact.burne._post_openai_response") as post:
        result = burne.run_burne_repair(
            _args(source_root=["services/api"], output="missing-key-report.json"),
            env=_env(OPENAI_API_KEY=""),
        )

    report = _load_report(output)
    assert result == 2
    assert report["status"] == "policy-failed"
    assert any(item["name"] == "OPENAI_API_KEY" for item in report["policy"]["violations"])
    post.assert_not_called()


def test_openai_timeout_returns_failed_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "timeout-report.json"

    with patch("pact.burne._post_openai_response", side_effect=TimeoutError):
        result = burne.run_burne_repair(
            _args(source_root=["services/api"], output="timeout-report.json"),
            env=_env(),
        )

    report = _load_report(output)
    assert result == 124
    assert report["accepted"] is False
    assert report["status"] == "timeout"
    assert report["agent"]["status"] == "timeout"


def test_failed_openai_request_returns_failed_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "failed-report.json"

    with patch("pact.burne._post_openai_response", side_effect=urllib.error.URLError("failed")):
        result = burne.run_burne_repair(
            _args(source_root=["services/api"], output="failed-report.json"),
            env=_env(),
        )

    report = _load_report(output)
    assert result == 1
    assert report["accepted"] is False
    assert report["status"] == "failed"
    assert report["agent"]["exit_code"] == 1


def test_incomplete_openai_response_fails_before_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "incomplete-report.json"
    handler = tmp_path / "services" / "api" / "handler.py"
    original_handler = handler.read_text(encoding="utf-8")

    with patch(
        "pact.burne._post_openai_response",
        return_value={
            "id": "resp_incomplete",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "status": "changed",
                                    "summary": "partial response",
                                    "changes": [{"path": "services/api/handler.py", "content": "owned\n"}],
                                }
                            )
                        }
                    ]
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        },
    ):
        result = burne.run_burne_repair(
            _args(source_root=["services/api"], output="incomplete-report.json"),
            env=_env(),
        )

    report = _load_report(output)
    assert result == 1
    assert report["accepted"] is False
    assert report["status"] == "failed"
    assert report["agent"]["applied_paths"] == []
    assert any(item["name"] == "openai-response" for item in report["policy"]["violations"])
    assert handler.read_text(encoding="utf-8") == original_handler


def test_write_guard_rejects_forbidden_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "write-guard-report.json"

    with patch(
        "pact.burne._post_openai_response",
        return_value=_response(
            {
                "status": "changed",
                "summary": "bad write",
                "changes": [{"path": ".github/workflows/owned.yml", "content": "name: bad\n"}],
            }
        ),
    ):
        result = burne.run_burne_repair(
            _args(source_root=["services/api"], output="write-guard-report.json"),
            env=_env(),
        )

    report = _load_report(output)
    assert result == 3
    assert report["accepted"] is False
    assert report["write_guard"]["status"] == "failed"
    assert ".github/workflows/owned.yml" in report["write_guard"]["forbidden_changed_paths"]
    assert not (tmp_path / ".github" / "workflows" / "owned.yml").exists()


def test_write_guard_rejects_symlinked_allowed_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    target = tmp_path / ".github" / "workflows" / "ci.yml"
    target.write_text("before\n", encoding="utf-8")
    (tmp_path / "services" / "api" / "ci-link.yml").symlink_to(target)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "symlink-report.json"

    with patch(
        "pact.burne._post_openai_response",
        return_value=_response(
            {
                "status": "changed",
                "summary": "bad symlink write",
                "changes": [{"path": "services/api/ci-link.yml", "content": "after\n"}],
            }
        ),
    ):
        result = burne.run_burne_repair(
            _args(source_root=["services/api"], output="symlink-report.json"),
            env=_env(),
        )

    report = _load_report(output)
    assert result == 3
    assert report["accepted"] is False
    assert report["write_guard"]["status"] == "failed"
    assert target.read_text(encoding="utf-8") == "before\n"
    assert any("symlinked write path" in item["message"] for item in report["policy"]["violations"])


def test_context_size_fails_closed_before_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "services" / "api" / "huge.py").write_text("x" * (burne.MAX_FILE_BYTES + 1), encoding="utf-8")

    with patch("pact.burne._post_openai_response") as post:
        result = burne.run_burne_repair(_args(source_root=["services/api"]), env=_env())

    assert result == 2
    post.assert_not_called()


def test_context_candidate_overflow_fails_closed_even_when_candidates_are_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    for index in range(burne.MAX_CONTEXT_FILES + 1):
        (tmp_path / "services" / "api" / f"binary-{index:03}.bin").write_bytes(b"\x00" * 4)

    with patch("pact.burne._post_openai_response") as post:
        result = burne.run_burne_repair(_args(source_root=["services/api"]), env=_env())

    assert result == 2
    assert post.call_count == 0
