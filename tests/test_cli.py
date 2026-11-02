from __future__ import annotations

import json
from pathlib import Path

import pytest

from cliphunter.cli import _config_overrides, build_parser, main


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["--file", "video.mp4"])
    assert args.file == "video.mp4"
    assert args.url is None
    assert args.live is False
    assert args.chat is None
    assert args.audio is None
    assert args.transcript is None
    assert args.vertical is None
    assert args.subtitles is None
    assert args.llm is None
    assert args.max_duration == 5.0


def test_parser_rejects_conflicting_signal_flags() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--file", "a.mp4", "--chat", "--no-chat"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--file", "a.mp4", "--vertical", "--no-vertical"])


def test_config_overrides_maps_every_option() -> None:
    args = build_parser().parse_args(
        [
            "--file",
            "video.mp4",
            "--clips",
            "3",
            "--min-dur",
            "10",
            "--max-dur",
            "40",
            "--pre-roll",
            "5",
            "--post-roll",
            "2",
            "--window",
            "12",
            "--hop",
            "3",
            "--no-chat",
            "--no-audio",
            "--transcript",
            "--model",
            "base",
            "--device",
            "cpu",
            "--language",
            "es",
            "--vertical",
            "--vertical-mode",
            "crop",
            "--subtitles",
            "--subtitle-mode",
            "chunk",
            "--llm",
            "--workers",
            "4",
            "--out",
            "salida",
            "--cache-dir",
            "cache",
        ]
    )
    overrides = _config_overrides(args)
    assert overrides["selection.clips"] == 3
    assert overrides["selection.min_duration"] == 10
    assert overrides["selection.max_duration"] == 40
    assert overrides["selection.pre_roll"] == 5
    assert overrides["selection.post_roll"] == 2
    assert overrides["analysis.window"] == 12
    assert overrides["analysis.hop"] == 3
    assert overrides["analysis.signals.chat"] is False
    assert overrides["analysis.signals.audio"] is False
    assert overrides["analysis.signals.transcript"] is True
    assert overrides["transcript.model"] == "base"
    assert overrides["transcript.device"] == "cpu"
    assert overrides["transcript.language"] == "es"
    assert overrides["export.vertical"] is True
    assert overrides["export.vertical_mode"] == "crop"
    assert overrides["subtitles.enabled"] is True
    assert overrides["subtitles.mode"] == "chunk"
    assert overrides["llm.enabled"] is True
    assert overrides["export.workers"] == 4
    assert overrides["paths.output_dir"] == "salida"
    assert overrides["paths.cache_dir"] == "cache"


def test_main_requires_exactly_one_source() -> None:
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        main(["--file", "a.mp4", "https://twitch.tv/videos/1"])
    assert info.value.code == 2


def test_main_rejects_live_with_local_file() -> None:
    with pytest.raises(SystemExit) as info:
        main(["--file", "a.mp4", "--live"])
    assert info.value.code == 2


def test_main_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert "ClipHunter" in capsys.readouterr().out


def test_main_dry_run_writes_analysis(
    tmp_path: Path, hype_video: Path
) -> None:
    out = tmp_path / "out"
    code = main(
        [
            "--file",
            str(hype_video),
            "--no-chat",
            "--no-transcript",
            "--clips",
            "2",
            "--min-dur",
            "5",
            "--max-dur",
            "15",
            "--dry-run",
            "--out",
            str(out),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert code == 0
    payload = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert payload["frames"] > 0
    assert payload["settings"]["min_duration"] == 5
    assert len(payload["clips"]) >= 1
    assert not (out / "clips.json").exists()


def test_main_exports_clips_and_manifest(
    tmp_path: Path, hype_video: Path
) -> None:
    out = tmp_path / "exported"
    code = main(
        [
            "--file",
            str(hype_video),
            "--no-chat",
            "--no-transcript",
            "--clips",
            "2",
            "--min-dur",
            "5",
            "--max-dur",
            "15",
            "--workers",
            "1",
            "--out",
            str(out),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert code == 0
    manifest = json.loads((out / "clips.json").read_text(encoding="utf-8"))
    assert manifest["tool"] == "ClipHunter"
    assert manifest["source"]["platform"] == "local"
    assert manifest["settings"]["signals"]["audio"] is True
    assert manifest["settings"]["signals"]["chat"] is False
    assert manifest["errors"] == []
    assert 1 <= len(manifest["clips"]) <= 2

    for clip in manifest["clips"]:
        target = out / clip["file"]
        assert target.is_file()
        assert target.stat().st_size > 0
        assert 5 <= clip["duration"] <= 15
        assert clip["signals"]


def test_main_applies_optional_llm(
    tmp_path: Path,
    hype_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cliphunter.llm as llm_module

    captured: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self) -> dict[str, Any]:
            content = json.dumps(
                {
                    "clips": [
                        {
                            "index": 1,
                            "score": 9,
                            "title": "Titulo del LLM",
                            "reason": "tiene gancho",
                        }
                    ]
                }
            )
            return {"choices": [{"message": {"content": content}}]}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured["payload"] = kwargs.get("json")
        return FakeResponse()

    monkeypatch.setenv("CLIPHUNTER_LLM_BASE_URL", "https://llm.local/v1")
    monkeypatch.setenv("CLIPHUNTER_LLM_MODEL", "mock-model")
    monkeypatch.setenv("CLIPHUNTER_LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_module.requests, "post", fake_post)

    out = tmp_path / "llm"
    code = main(
        [
            "--file",
            str(hype_video),
            "--no-chat",
            "--no-transcript",
            "--clips",
            "1",
            "--min-dur",
            "5",
            "--max-dur",
            "15",
            "--llm",
            "--workers",
            "1",
            "--out",
            str(out),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert code == 0
    assert captured["url"] == "https://llm.local/v1/chat/completions"
    assert captured["payload"]["model"] == "mock-model"

    manifest = json.loads((out / "clips.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["llm"] is True
    clip = manifest["clips"][0]
    assert clip["title"] == "Titulo del LLM"
    assert clip["llm_score"] == 9
    assert clip["llm_reason"] == "tiene gancho"
