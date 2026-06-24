"""SRT generation test.

Guarded for the full GPU/api venv: importing the pipeline pulls in pyrubberband
and silero_vad at module top, which are absent in the lean test env, so this
skips cleanly here and runs in production.
"""
import re
import pytest

pytest.importorskip("pyrubberband")
pytest.importorskip("silero_vad")

from langswap.translation_pipeline import VideoTranslationPipeline  # noqa: E402
from langswap.file_repository import LocalOnlyFileRepository  # noqa: E402
from langswap.pipeline_models.models import (  # noqa: E402
    VideoTranslation,
    TextedSegment,
    TranslatedTextedSegment,
)

VTT_TIME = r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}"


def test_generate_vtt_files(tmp_path):
    repo = LocalOnlyFileRepository("job1", str(tmp_path))

    recognized = [
        TextedSegment(text="Hello world", start=0.0, end=1.5, speaker="SPEAKER_00"),
        TextedSegment(text="How are you", start=2.0, end=3.25, speaker="SPEAKER_00"),
    ]
    translated = [
        TranslatedTextedSegment(
            text="Hello world", start=0.0, end=1.5, translation="Privet mir",
            source_file=None, generated_file=None, speaker="SPEAKER_00",
        ),
        TranslatedTextedSegment(
            text="How are you", start=2.0, end=3.25, translation="Kak dela",
            source_file=None, generated_file=None, speaker="SPEAKER_00",
        ),
    ]

    pipeline = object.__new__(VideoTranslationPipeline)
    pipeline._file_repository = repo
    pipeline.video_translation = VideoTranslation(
        public_id="job1",
        recognized_texts=recognized,
        translated_texts=translated,
    )

    source_vtt, translated_vtt = pipeline.generate_vtt_files()

    source_text = open(source_vtt.file_path, encoding="utf-8").read()
    translated_text = open(translated_vtt.file_path, encoding="utf-8").read()

    assert source_vtt.file_path.endswith(".vtt")
    assert translated_vtt.file_path.endswith(".vtt")

    assert source_text.startswith("WEBVTT\n\n")
    assert translated_text.startswith("WEBVTT\n\n")

    assert re.search(VTT_TIME, source_text)
    assert "00:00:00.000 --> 00:00:01.500" in source_text
    assert "00:00:02.000 --> 00:00:03.250" in source_text

    assert "<v SPEAKER_00>Hello world</v>" in source_text
    assert "<v SPEAKER_00>How are you</v>" in source_text
    assert "<v SPEAKER_00>Privet mir</v>" in translated_text
    assert "<v SPEAKER_00>Kak dela</v>" in translated_text

    assert not any(line.strip().isdigit() for line in source_text.splitlines())
    assert not any(line.strip().isdigit() for line in translated_text.splitlines())
