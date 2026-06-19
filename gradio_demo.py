"""Gradio demo: run the langswap video translation pipeline locally (no S3).

Launch with:
    python gradio_demo.py

Or with custom options:
    python gradio_demo.py --port 7860 --share

Uses LocalOnlyFileRepository so no AWS credentials are needed.  All
intermediate artifacts and the final dubbed video are written under
``data/<public_id>/`` relative to the working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

# Load .env (HF_TOKEN, ELEVEN_API_KEY, …) before anything else reads the
# environment.
load_dotenv()

from langswap import backends
from langswap.file_repository import LocalOnlyFileRepository
from langswap.pipeline_models.models import TranslationPipelineConfig
from langswap.translation_pipeline import VideoTranslationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("langswap.gradio")

BASE_DIR = os.environ.get("LANGSWAP_DATA_DIR", "data")

LANGUAGES = ["english", "russian", "spanish", "french", "german", "italian",
             "portuguese", "polish", "dutch", "turkish", "arabic", "hindi",
             "japanese", "korean", "chinese"]

# Backend choices come from langswap/backends.json (single source of truth).
TTS_ENGINES = backends.options("tts")
ASR_BACKENDS = backends.options("asr")
TRANSLATION_BACKENDS = backends.options("translation")
DUBBING_ALGOS = backends.options("dubbing")


def _public_id(video_path: str) -> str:
    """Derive a stable ID from the input path so caches survive re-runs."""
    return hashlib.md5(str(video_path).encode("utf-8")).hexdigest()[:12]


def _resolve_device(requested: str) -> str:
    """Map a user-friendly device label to what the pipeline expects."""
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def translate_video(
    video_file: str,
    source_language: str,
    target_language: str,
    tts_engine: str,
    asr_backend: str,
    translation_backend: str,
    dubbing_algo: str,
    device: str,
    skip_diarization: bool,
    eleven_api_token: Optional[str],
    # track_tqdm=False: show only our explicit stage progress below, not every
    # library's internal tqdm (huggingface_hub's cache-check bars would otherwise
    # surface as confusing "Downloading 0/0 B" bars that aren't real downloads).
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Gradio callback: run the full pipeline and return outputs.

    Returns a tuple (result_video, source_vtt, translated_vtt, status_text).
    Any intermediate failure is caught and reported via the status field so
    the UI does not just crash with a stack trace.
    """
    if not video_file:
        return None, None, None, "No video file uploaded."

    try:
        video_path = Path(video_file).expanduser().resolve()
        if not video_path.exists():
            return None, None, None, f"File not found: {video_path}"

        public_id = _public_id(str(video_path))
        repo = LocalOnlyFileRepository(public_id, BASE_DIR)
        resolved_device = _resolve_device(device)

        # ElevenLabs client reads ELEVEN_API_KEY from the environment; honor a
        # token entered in the UI by exporting it for this process.
        if eleven_api_token:
            os.environ["ELEVEN_API_KEY"] = eleven_api_token

        progress(0.0, desc="Initializing pipeline")

        config = TranslationPipelineConfig(
            source_lang=source_language or None,
            target_lang=target_language,
            name=public_id,
            public_id=public_id,
            num_speakers=None,
            source_video_path=str(video_path),
            base_dir=BASE_DIR,
            device=resolved_device,
            tts_model=tts_engine,
            dubbing_algo=dubbing_algo,
            watermark=False,
            skip_diarization=skip_diarization,
            asr_backend=asr_backend,
            translation_backend=translation_backend,
        )

        pipeline = VideoTranslationPipeline(config=config, file_repository=repo)

        progress(0.15, desc="Speech-to-Text (ASR)")
        pipeline._generate_asr()

        progress(0.40, desc="Translating")
        pipeline._generate_translation()

        progress(0.60, desc="Text-to-Speech synthesis")
        pipeline._generate_speech()

        progress(0.80, desc="Mixing & dubbing")
        video_translation = pipeline._merge(pipeline.config.dubbing_algo)
        pipeline.video_translation = video_translation

        progress(0.95, desc="Writing subtitles")
        source_vtt, translated_vtt = pipeline.generate_vtt_files()

        progress(1.0, desc="Done")

        result_video_path = video_translation.processed_video.file_path
        status = (
            f"Success on device={resolved_device}.\n"
            f"Output dir: {repo.directory}\n"
            f"Video: {result_video_path}"
        )
        return result_video_path, source_vtt.file_path, translated_vtt.file_path, status

    except Exception as e:  # noqa: BLE001 — display to user, do not crash UI
        tb = traceback.format_exc()
        logger.exception("Pipeline failure")
        return None, None, None, f"Pipeline failed: {e}\n\n{tb}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="langswap — local video translation demo") as demo:
        gr.Markdown(
            "# langswap — local video translation demo\n"
            "Upload a video, pick source & target languages, and run the "
            "translation pipeline entirely on this machine. No S3 required."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Input video", sources=["upload"])

                with gr.Row():
                    source_lang = gr.Dropdown(
                        LANGUAGES, value="russian", label="Source language",
                        allow_custom_value=True,
                    )
                    target_lang = gr.Dropdown(
                        LANGUAGES, value="english", label="Target language",
                        allow_custom_value=True,
                    )

                with gr.Accordion("Models / backends", open=False):
                    tts_engine = gr.Dropdown(TTS_ENGINES, value=backends.default("tts"), label="TTS engine")
                    asr_backend = gr.Dropdown(ASR_BACKENDS, value=backends.default("asr"), label="ASR backend")
                    translation_backend = gr.Dropdown(
                        TRANSLATION_BACKENDS, value=backends.default("translation"), label="Translation backend"
                    )
                    dubbing_algo = gr.Dropdown(
                        DUBBING_ALGOS, value=backends.default("dubbing"), label="Dubbing algorithm"
                    )
                    device = gr.Dropdown(
                        ["auto", "cuda", "mps", "cpu"], value="auto", label="Device"
                    )
                    skip_diarization = gr.Checkbox(
                        value=True,
                        label="Skip speaker diarization (faster; recommended when pyannote weights are unavailable)",
                    )
                    eleven_token = gr.Textbox(
                        label="ElevenLabs API token (only for tts_engine=elevenlabs)",
                        type="password",
                        value="",
                    )

                run_btn = gr.Button("Translate", variant="primary")

            with gr.Column(scale=1):
                video_out = gr.Video(label="Translated video")
                source_srt_out = gr.File(label="Source transcript (.vtt)")
                target_srt_out = gr.File(label="Translated transcript (.vtt)")
                status_out = gr.Textbox(label="Status", lines=10)

        run_btn.click(
            fn=translate_video,
            inputs=[
                video_in, source_lang, target_lang, tts_engine, asr_backend,
                translation_backend, dubbing_algo, device, skip_diarization,
                eleven_token,
            ],
            outputs=[video_out, source_srt_out, target_srt_out, status_out],
        )

        gr.Markdown(
            "### Tips\n"
            "- First run auto-downloads the model weights into `models_weights/` — "
            "this can take a long time and needs free disk space.\n"
            "- Intermediate artifacts are cached under `data/<id>/` — re-running on "
            "the same file skips ASR/translation if their JSON outputs already exist."
        )

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the langswap Gradio demo.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the local data directory used for intermediate files.",
    )
    args = parser.parse_args()

    if args.data_dir:
        global BASE_DIR
        BASE_DIR = args.data_dir
        os.environ["LANGSWAP_DATA_DIR"] = args.data_dir

    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

    demo = build_ui()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
