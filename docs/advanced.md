# langswap — advanced guide

Detailed reference for running langswap from source, the model registry, all
environment variables, Docker build notes, and troubleshooting.
For the quick path see the [README](../README.md).

The pipeline: **ASR** (faster-whisper large-v3 + Silero VAD) → **translation** (Gemma-4-E2B) →
**TTS** (OmniVoice) → **dubbing/merge** → muxed video + SRT subtitles.
It runs entirely on a local machine (no S3/AWS required).

---

## 1. Prerequisites

- **Python 3.12** and [`uv`](https://github.com/astral-sh/uv)
- **NVIDIA GPU** with a recent driver (CUDA 13 capable; the stack uses `torch==2.11.0+cu130`)
- **System tools:**
  - `ffmpeg` — audio/video processing
  - `rubberband-cli` — time-stretching for the `speedup` / `stretch_whole` dubbing algorithms
    ```bash
    sudo apt-get install -y ffmpeg rubberband-cli
    ```
- **Docker** + the **NVIDIA Container Toolkit** — to run the all-in-one container
- **HuggingFace token** (`HF_TOKEN`) — required only for the gated `pyannote/speaker-diarization-3.1` model (used when diarization is enabled). The default backends' weights download without a token.

Create a `.env` in the project root:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
ELEVEN_API_KEY=...                     # only if using the ElevenLabs TTS backend
MODEL_WEIGHTS_DIR=./models_weights     # where model weights live (used by Docker too)
LANGSWAP_DATA_DIR=./data               # where outputs/artifacts go
```

---

## 2. Install dependencies

**uv is the only supported installer, and `pyproject.toml` is the single source of
truth for dependencies** (there is no `requirements.txt` / lock file). One command
per variant:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[gpu]"          # full local-model stack (NVIDIA GPU)
# or, Mac / no GPU — hosted APIs, far fewer deps:
uv pip install -e ".[api]"
# RunPod serverless worker (what the Docker image installs):
uv pip install -e ".[gpu,runpod]"
```

No pre-install step or index flags are needed: `torch`/`torchaudio`/`torchcodec`
for the `[gpu]` extra resolve from the CUDA 13 wheel index, and `llama-cpp-python`
from its prebuilt CPU wheel — both wired in pyproject's `[tool.uv]` section. The
`[api]` extra's torch resolves from PyPI (CPU/MPS). `transformers==5.9.0` /
`vllm==0.21.0` are pinned in the `[gpu]` extra to the validated ABI pair OmniVoice
needs, so torch is resolved to a compatible cu130 build automatically.

---

## 3. Model weights

Models are loaded directly from HuggingFace and **auto-downloaded on first use** into the
project's `models_weights/` directory (i.e. `MODEL_WEIGHTS_DIR`, which defaults to
`./models_weights`). Keeping weights in the project tree is what makes the container and runpod
builds self-contained. Set `HF_TOKEN` for the gated models.

| Model            | Default repo / id                                                  | Gated   | Used by                    |
|------------------|--------------------------------------------------------------------|---------|----------------------------|
| faster-whisper   | `large-v3`                                                         | No      | ASR `vad` (default)        |
| Silero VAD       | `silero-vad` (bundled, ~1.4 MB)                                    | No      | ASR `vad` segmentation     |
| Gemma-4-E2B GGUF | `unsloth/gemma-4-E2B-it-GGUF` / `gemma-4-E2B-it-UD-Q4_K_XL.gguf`   | No      | translation `llamacpp` (default) |
| OmniVoice        | `k2-fsa/OmniVoice`                                                 | No      | TTS `omnivoice` (default)  |
| pyannote         | `pyannote/speaker-diarization-3.1`                                | **Yes** | diarization (optional)     |

Model ids and repos are **hardcoded** in each client (no env-var overrides). To use a
different model, change the constant in the relevant `*_client.py`. The hosted-API backends
(`openai` ASR/translation, `elevenlabs` TTS) download no weights.

---

## 4. Run locally for debugging (`main.py local`)

Runs each pipeline stage separately with verbose logging and caches intermediate JSON under
`data/<id>/`, so reruns skip stages that already succeeded.

```bash
.venv/bin/python main.py local 12.mp4 english russian 2>&1 | tee /tmp/langswap_debug.log
```

Positional args: `local <video> [target_lang] [source_lang]` (source is auto-detected if omitted).

Useful flags:

| Flag                  | Default     | Notes                                                       |
|-----------------------|-------------|-------------------------------------------------------------|
| `--device`            | `auto`      | `auto` / `cuda` / `mps` / `cpu`                             |
| `--asr`               | `vad`       | `vad` (faster-whisper + Silero VAD) / `openai` (Whisper API) |
| `--translation`       | `llamacpp`  | `llamacpp` (Gemma-4-E2B GGUF) / `openai`                   |
| `--tts`               | `omnivoice` | `omnivoice` / `elevenlabs`                                 |
| `--dubbing`           | `speedup`   | `speedup` / `stretch_whole` / `pause_based`                |
| `--with-diarization`  | off         | enable speaker diarization (needs pyannote weights + token) |
| `--stop-after`        | —           | stop after `asr` / `translation` / `tts` / `merge` / `srt` / `vtt`  |

Output: `data/<id>/resulted_video.mp4`, plus `source_transcript.srt`, `translated_transcript.srt`, `source_transcript.vtt`, and `translated_transcript.vtt`.

---

## 5. Run the Gradio demo (`gradio_demo.py`)

Browser UI for the full pipeline; uses `LocalOnlyFileRepository` (no AWS/S3).

```bash
.venv/bin/python gradio_demo.py          # http://localhost:7860  (add --share for a public link)
```

It loads `.env` and runs the full pipeline locally with the default backends
(`vad` ASR + `llamacpp` translation + `omnivoice` TTS), all in-process. Switch backends /
languages / dubbing algorithm in the *Models / backends* accordion.

Other flags: `--host`, `--port`, `--share`, `--data-dir`.

---

## 6. Docker build notes

The single [`Dockerfile`](../Dockerfile) builds the whole pipeline into one image on
`transformers==5.9.0` + `vllm==0.21.0` (the pair vllm-omni's voice cloning requires). The default
path is `vad` ASR (faster-whisper large-v3 + Silero VAD) + Gemma-4-E2B GGUF translation + OmniVoice
TTS, all in-process — no separate ASR/TTS service.

- Built on `ubuntu24.04` (Python 3.12 is native there; 22.04 only ships 3.10).
- **cuBLAS-12 for ctranslate2:** this is a CUDA-13 image (`cu130` torch/vLLM/OmniVoice), but
  ctranslate2 (faster-whisper's backend) links cuBLAS **12** (`libcublas.so.12`). The Dockerfile
  installs `nvidia-cublas-cu12` alongside the cu13 stack so GPU ASR works without overwriting the
  cu13 cuDNN that torch/vLLM rely on.
- `demucs` is built from a git sdist, which fetches `setuptools`/`wheel` from PyPI during build
  isolation. The Dockerfile wraps the dependency install in a retry loop and installs the editable
  package with `--no-build-isolation` to tolerate flaky outbound network.
- You will see `vLLM and vLLM-Omni appear to have mismatched major/minor versions` — this warning
  is expected for the locked `vllm 0.21.0` / `vllm-omni 0.20.0` pair and is harmless.

---

## Environment variables

| Variable                      | Purpose                                                            |
|-------------------------------|--------------------------------------------------------------------|
| `HF_TOKEN`                    | HuggingFace token for the gated `pyannote` diarization model       |
| `OPENAI_API_KEY`              | OpenAI key (only for `--asr openai` / `--translation openai`)      |
| `ELEVEN_API_KEY`              | ElevenLabs key (only for `--tts elevenlabs`)                       |
| `MODEL_WEIGHTS_DIR`           | Where weights are stored/loaded (default `./models_weights`)       |
| `LANGSWAP_DATA_DIR`           | Where intermediate artifacts/outputs go (default `data/`)          |

There are intentionally no `LANGSWAP_*` model/tuning knobs — model ids, GPU offload, and warm
reuse are all hardcoded to sensible defaults; change the constant in the relevant client to adjust.

---

## Troubleshooting

- **`undefined symbol: ...getCurrentCUDABlasHandle`** — torch is newer than vLLM expects; pin
  `torch==2.11.0+cu130` (see §2).
- **`Library libcublas.so.12 is not found`** — faster-whisper's ctranslate2 backend needs cuBLAS 12,
  which the CUDA-13 stack doesn't ship. Install `nvidia-cublas-cu12` (the Dockerfile already does).
- **`Failed to execute rubberband`** — install `rubberband-cli` (needed by `speedup`/`stretch_whole`).
- **`Cannot re-initialize CUDA in forked subprocess`** — handled in code (vLLM uses `spawn`); if you
  hit it elsewhere, set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- **`Free memory ... less than desired GPU memory utilization`** — the GPU is shared (OmniVoice +
  translation + ASR are co-resident via warm reuse). Free other processes, or use the API backends
  on a small GPU. Note Gradio is long-lived, so restart it to release accumulated GPU memory
  between runs.

---

## Project structure

- `main.py` — CLI entrypoint (`local` stage-by-stage runner + `runpod` S3 job runner)
- `serverless.py` — RunPod serverless handler
- `gradio_demo.py` — local browser UI
- `Dockerfile` — all-in-one image (full pipeline + Gradio UI on transformers 5.9 / vllm 0.21)
- `langswap/translation_pipeline.py` — pipeline orchestration
- `langswap/ml/` — ASR / translation / TTS / dubbing implementations: each service has a `*_manager.py`
  that dispatches to one client `.py` per backend
- `langswap/backends.json` — selectable ASR / translation / TTS / dubbing backends (UI reads this only)
- `langswap/model_config.py` — points `MODEL_WEIGHTS_DIR` and the HF/torch/vLLM cache dirs at the weights dir
- `scripts/` — dev/ops helpers (batch translate, quick check, RunPod integration, Docker build)

---

## Model licenses

This repository licenses the **pipeline code only** ([AGPL-3.0-or-later](../LICENSE)). The models it
downloads and runs at inference time are **not** covered by the AGPL and carry their own terms, some
of which restrict commercial or derivative use:

| Component | Model | License |
| --- | --- | --- |
| Translation | Gemma-4-E2B-IT (Google **Gemma**, `unsloth` GGUF) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — *use restrictions apply; not OSI-approved* |
| ASR | faster-whisper `large-v3` (OpenAI Whisper) + Silero VAD | Whisper [MIT](https://github.com/openai/whisper/blob/main/LICENSE); Silero VAD [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) |
| TTS | `k2-fsa/OmniVoice` | See the model's Hugging Face model card |
| ElevenLabs / OpenAI backends | Hosted APIs | Provider commercial Terms of Service |

You are responsible for complying with each model's license for your use case. The AGPL grant on this
code does **not** grant any rights to the model weights.
