from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cliphunter.config import load_config
from cliphunter.exceptions import ClipHunterError


def test_defaults_load() -> None:
    config = load_config()
    assert config.i("selection.clips") == 10
    assert config.f("analysis.window") == 10.0
    assert config.b("analysis.signals.chat") is True
    assert config.section("weights")["audio"] > 0


def test_user_yaml_merges_nested_values(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(
        yaml.safe_dump({"selection": {"clips": 3}, "export": {"vertical": True}}),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.i("selection.clips") == 3
    assert config.b("export.vertical") is True
    assert config.f("selection.min_duration") == 15.0


def test_env_and_cli_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIPHUNTER_WHISPER_MODEL", "base")
    config = load_config()
    assert config.get("transcript.model") == "base"

    config = load_config(overrides={"transcript.model": "tiny"})
    assert config.get("transcript.model") == "tiny"


def test_invalid_configurations_raise() -> None:
    with pytest.raises(ClipHunterError):
        load_config(overrides={"selection.min_duration": 60, "selection.max_duration": 10})
    with pytest.raises(ClipHunterError):
        load_config(overrides={"selection.clips": 0})
    with pytest.raises(ClipHunterError):
        load_config(overrides={"analysis.hop": 0})
    with pytest.raises(ClipHunterError):
        load_config(overrides={"weights": {"chat": 0, "audio": 0, "transcript": 0}})


def test_snapshot_redacts_api_key() -> None:
    config = load_config(overrides={"llm.api_key": "super-secret"})
    assert config.snapshot()["llm"]["api_key"] == "***"


def test_config_hash_changes_with_section() -> None:
    first = load_config()
    second = load_config(overrides={"analysis.window": 12})
    assert first.hash("analysis") != second.hash("analysis")
    assert first.hash("export") == second.hash("export")
