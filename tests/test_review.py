"""Tests for provider-neutral Advocate and Simulacrum review orchestration."""

from __future__ import annotations

import sys
from importlib.resources import files
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pact.review import (
    _tool_env,
    find_simulacrum_command,
    render_review_summary,
    run_reviews,
    select_advocate_provider,
)


def test_find_simulacrum_command_uses_explicit_configuration():
    with patch.dict("os.environ", {"PACT_SIMULACRUM_CMD": "python3 /tmp/sim.py"}):
        assert find_simulacrum_command() == ["python3", "/tmp/sim.py"]


def test_find_simulacrum_command_uses_packaged_runtime_by_default():
    with patch.dict("os.environ", {}, clear=True):
        assert find_simulacrum_command() == [
            sys.executable,
            "-m",
            "pact.vendor.simulacrum",
        ]


def test_packaged_simulacrum_includes_annotated_corpus():
    corpus = files("pact.vendor.simulacrum").joinpath(
        "adversarial_pairs_annotated.json"
    )
    assert corpus.is_file()
    assert '"annotation"' in corpus.read_text()


def test_packaged_simulacrum_routes_to_specialist_without_openai():
    from pact.vendor.simulacrum import runtime

    fake_anthropic = SimpleNamespace(
        Anthropic=lambda api_key: SimpleNamespace(api_key=api_key)
    )
    with patch.dict(
        "sys.modules",
        {"anthropic": fake_anthropic, "openai": None},
    ), patch.dict(
        "os.environ",
        {"ANTHROPIC_API_KEY": "test-key"},
        clear=True,
    ), patch.object(
        runtime,
        "GENERALIST_MODEL",
        "",
    ), patch.object(
        runtime,
        "_classify_phase",
        return_value=("GENERALIST", "recall question"),
    ), patch.object(
        runtime,
        "_classify_mode_a",
        return_value=("DEFAULT", ""),
    ), patch.object(
        runtime,
        "_build_specialist_prompt",
        return_value="system",
    ), patch.object(
        runtime,
        "_call_specialist",
        return_value="specialist answer",
    ):
        result = runtime.call_simulacrum("What is Pact?", None)

    assert result["text"] == "specialist answer"
    assert result["phase"] == "SPECIALIST"
    assert "Generalist branch is disabled" in result["phase_reason"]


def test_tool_env_uses_per_process_fallback_key():
    with patch.dict(
        "os.environ",
        {"PACT_REVIEW_ANTHROPIC_API_KEY": "review-key"},
        clear=True,
    ):
        env = _tool_env()
    assert env["ANTHROPIC_API_KEY"] == "review-key"


def test_select_advocate_provider_uses_standard_credentials():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "openai-key"}, clear=True):
        assert select_advocate_provider() == "openai"


def test_run_reviews_persists_reports(tmp_path: Path):
    target = tmp_path / "repo"
    target.mkdir()
    output = tmp_path / "reviews"

    def fake_run(command, **kwargs):
        if command[0] == "/usr/local/bin/advocate":
            artifact = Path(command[command.index("--output") + 1])
            artifact.write_text('{"findings": []}')
            return subprocess.CompletedProcess(command, 0, "advocate ok", "")
        return subprocess.CompletedProcess(command, 0, "framing held", "")

    with patch("pact.review.shutil.which", return_value="/usr/local/bin/advocate"), \
         patch("pact.review.find_simulacrum_command", return_value=["/tmp/sim.py"]), \
         patch("pact.review.subprocess.run", side_effect=fake_run):
        report = run_reviews(
            target,
            claim="This change is done because all contract tests pass.",
            output_dir=output,
        )

    assert report["ok"] is True
    assert (output / "advocate.json").exists()
    assert (output / "simulacrum.md").read_text() == "framing held"
    assert (output / "review.json").exists()
    assert report["simulacrum"]["status"] == "completed"
    assert "Review tools: completed" in render_review_summary(report)


def test_run_reviews_requires_claim_for_simulacrum(tmp_path: Path):
    target = tmp_path / "repo"
    target.mkdir()

    report = run_reviews(
        target,
        run_advocate=False,
        run_simulacrum=True,
        output_dir=tmp_path / "reviews",
    )

    assert report["ok"] is False
    assert report["simulacrum"]["status"] == "failed"
    assert "--claim is required" in report["simulacrum"]["message"]


def test_run_reviews_surfaces_simulacrum_failure_stderr(tmp_path: Path):
    target = tmp_path / "repo"
    target.mkdir()

    with patch(
        "pact.review.subprocess.run",
        return_value=subprocess.CompletedProcess(
            ["python", "-m", "pact.vendor.simulacrum"],
            1,
            "",
            "Set ANTHROPIC_API_KEY for the specialist + classifier.",
        ),
    ):
        report = run_reviews(
            target,
            claim="The frame holds.",
            run_advocate=False,
            output_dir=tmp_path / "reviews",
        )

    assert report["ok"] is False
    assert report["simulacrum"]["status"] == "failed"
    assert "Set ANTHROPIC_API_KEY" in report["simulacrum"]["message"]


def test_run_reviews_can_run_advocate_only(tmp_path: Path):
    target = tmp_path / "repo"
    target.mkdir()

    def fake_run(command, **kwargs):
        artifact = Path(command[command.index("--output") + 1])
        artifact.write_text('{"persona_reports": []}')
        return subprocess.CompletedProcess(command, 0, "ok", "")

    with patch("pact.review.shutil.which", return_value="/usr/local/bin/advocate"), \
         patch("pact.review.subprocess.run", side_effect=fake_run):
        report = run_reviews(
            target,
            run_simulacrum=False,
            output_dir=tmp_path / "reviews",
        )

    assert report["ok"] is True
    assert report["advocate"]["status"] == "passed"
    assert report["simulacrum"]["status"] == "not_requested"


def test_run_reviews_fails_when_advocate_personas_fail(tmp_path: Path):
    target = tmp_path / "repo"
    target.mkdir()

    def fake_run(command, **kwargs):
        artifact = Path(command[command.index("--output") + 1])
        artifact.write_text(
            '{"persona_reports": ['
            '{"persona": "red_team", "summary": "Error: auth failed"}'
            ']}'
        )
        return subprocess.CompletedProcess(command, 0, "looks fine", "")

    with patch("pact.review.shutil.which", return_value="/usr/local/bin/advocate"), \
         patch("pact.review.subprocess.run", side_effect=fake_run):
        report = run_reviews(
            target,
            run_simulacrum=False,
            output_dir=tmp_path / "reviews",
        )

    assert report["ok"] is False
    assert report["advocate"]["status"] == "failed"
    assert "red_team" in report["advocate"]["message"]


def test_run_reviews_fails_on_blocking_advocate_finding(tmp_path: Path):
    target = tmp_path / "repo"
    target.mkdir()

    def fake_run(command, **kwargs):
        artifact = Path(command[command.index("--output") + 1])
        artifact.write_text(
            '{"persona_reports": [{"persona": "sage", "summary": "reviewed", '
            '"findings": [{"severity": "high", "title": "Broken boundary"}]}]}'
        )
        return subprocess.CompletedProcess(command, 0, "reviewed", "")

    with patch("pact.review.shutil.which", return_value="/usr/local/bin/advocate"), \
         patch("pact.review.subprocess.run", side_effect=fake_run):
        report = run_reviews(
            target,
            run_simulacrum=False,
            output_dir=tmp_path / "reviews",
        )

    assert report["ok"] is False
    assert report["advocate"]["status"] == "failed"
    assert "Broken boundary" in report["advocate"]["message"]


def test_run_reviews_rejects_non_positive_timeout(tmp_path: Path):
    target = tmp_path / "repo"
    target.mkdir()

    try:
        run_reviews(target, timeout=0)
    except ValueError as exc:
        assert "greater than zero" in str(exc)
    else:
        raise AssertionError("Expected non-positive timeout to be rejected")
