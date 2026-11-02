from __future__ import annotations

from pathlib import Path

from cliphunter.config import load_config
from cliphunter.exporter import build_filters, clip_filename, cut_segment
from cliphunter.models import CandidateClip
from cliphunter.utils import probe_media


def test_build_filters_vertical_blur_uses_background_copy() -> None:
    parts, label = build_filters(
        vertical=True,
        vertical_mode="blur",
        blur=24,
        brightness=-0.05,
        subtitle_name=None,
        subtitle_resolution=None,
    )
    assert label == "vbase"
    assert any("boxblur=24:2" in part for part in parts)
    assert any("overlay=(W-w)/2:(H-h)/2" in part for part in parts)


def test_build_filters_vertical_crop_and_subtitles() -> None:
    parts, label = build_filters(
        vertical=True,
        vertical_mode="crop",
        blur=20,
        brightness=0.0,
        subtitle_name="clip.ass",
        subtitle_resolution=(1080, 1920),
    )
    assert label == "vout"
    assert any("crop=1080:1920" in part for part in parts)
    assert any(part.startswith("[vbase]ass=") for part in parts)


def test_build_filters_horizontal_without_effects() -> None:
    parts, label = build_filters(
        vertical=False,
        vertical_mode="blur",
        blur=20,
        brightness=0.0,
        subtitle_name=None,
        subtitle_resolution=None,
    )
    assert parts == []
    assert label is None


def test_clip_filename_slugifies_unicode_title() -> None:
    clip = CandidateClip(index=3, start=0.0, end=15.0, peak=1.0, score=0.5, title="¡Qué locura! 100%")
    target = clip_filename(clip, Path("clips"))
    assert target.name.startswith("clip_003_")
    assert target.suffix == ".mp4"
    assert "%" not in target.name
    assert "que-locura" in target.name


def test_cut_segment_exports_horizontal_clip(tmp_path: Path, tiny_video: Path) -> None:
    config = load_config()
    output = tmp_path / "clips" / "clip.mp4"
    cut_segment(tiny_video, 2.0, 7.0, output, config=config, title="Test clip", log=False)
    assert output.is_file()
    info = probe_media(output)
    assert 4.5 <= info.duration <= 5.6
    assert info.has_audio


def test_cut_segment_exports_vertical_clip_with_subtitles(tmp_path: Path, tiny_video: Path) -> None:
    from cliphunter.subtitles import build_ass
    from cliphunter.models import TranscriptData, TranscriptSegment, Word

    config = load_config()
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    transcript = TranscriptData(
        words=[Word(2.2, 2.8, "hola"), Word(2.9, 3.6, "mundo")],
        segments=[TranscriptSegment(2.2, 3.6, "hola mundo")],
    )
    ass = build_ass(transcript, 2.0, 7.0, out_dir / "clip.ass", config=config, vertical=True, width=1080, height=1920)
    assert ass is not None
    output = out_dir / "clip.mp4"
    cut_segment(
        tiny_video,
        2.0,
        7.0,
        output,
        config=config,
        vertical=True,
        subtitle_path=ass,
        title="Vertical",
        log=False,
    )
    info = probe_media(output)
    assert (info.width, info.height) == (1080, 1920)
    assert 4.5 <= info.duration <= 5.6
