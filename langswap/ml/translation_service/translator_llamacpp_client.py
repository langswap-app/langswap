"""Gemma-4-E2B translation client backed by llama-cpp-python + a GGUF model.

Why E2B + an isochrony loop: the dubbing image already carries vLLM (OmniVoice
needs it); a second vLLM engine for translation would split GPU memory and risk
OOM. llama-cpp-python loads a small Q4 GGUF in ~2 s and runs in-process. Gemma-4
E2B (~2B effective) is fast and clean but its single-shot length match is loose
(isochrony ~0.91). A short Python length-feedback loop — generate, measure
spoken length, re-prompt longer/shorter — tightens the match (~3-4x lower error)
for a few cents of extra latency, which matters because the downstream dub retimes
to the source segment duration.

Vision encoder: we load ONLY the text GGUF. Gemma-4's vision tower lives in a
separate `mmproj` file that we never pass and llama.cpp never instantiates, so
the multimodal encoder is absent from the process — exactly what we want for a
text-only translation task.

Single-GPU pinning: we load with split_mode=NONE and main_gpu=0 so llama.cpp
uses exactly one GPU. On a multi-GPU host, GGML otherwise registers a backend
per visible CUDA device, and Gemma-4's SWA graph then exceeds
GGML_SCHED_MAX_SPLIT_INPUTS when the scheduler tries to split it across
backends. Pinning to one device makes the multi-GPU case behave like the
single-GPU RunPod target (which never hit this), at no speed cost for a ~2B
model that fits on one card anyway.

Interface matches the other translator clients: load_models() + translate().
"""
from __future__ import annotations

import logging
import os
import unicodedata
from typing import List, Optional

from langswap.model_config import MODEL_WEIGHTS_DIR
from langswap.utils.ml_processing.lang2code_mapper import map_language_to_code

logger = logging.getLogger(__name__)

# Gemma-4-E2B instruction-tuned GGUF (Unsloth Dynamic 4-bit, ~3.2 GB), text-only.
_GGUF_REPO = "unsloth/gemma-4-E2B-it-GGUF"
_GGUF_FILE = "gemma-4-E2B-it-UD-Q4_K_XL.gguf"
_GGUF_DIR = "gemma-4-E2B-it-GGUF"

# Isochrony loop, intentionally forgiving: stop as soon as the spoken length is
# within +-15% of the source, and never iterate more than a couple of times — the
# point is a rough timing match, not to bully the model into padding/truncating
# (which costs meaning, see the guard in _translate_one).
_ISO_TOL = 0.15
_ISO_MAX_ITERS = 2

# Vowels used as a syllable proxy for alphabetic scripts (Latin + Cyrillic).
_VOWELS = set("aeiouy") | set("аеёиоуыэюя")


def _lang_name(lang: str) -> str:
    """Best-effort human-readable language name for the prompt."""
    l = (lang or "").strip()
    if not l:
        return l
    if len(l) == 2 and l.isalpha():
        try:
            return map_language_to_code(l.lower(), system="reverse_from_whisper").capitalize()
        except Exception:
            return l
    return l.capitalize()


def _spoken_length(text: str) -> int:
    """Rough spoken-length (syllable) proxy used to drive the isochrony loop.

    Counts vowels for alphabetic scripts (diacritics folded via NFKD so é→e);
    falls back to CJK/Kana/Hangul code points (~1 syllable each), then to the
    non-space character count. Always >= 1.
    """
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(c)
    )
    n = sum(c in _VOWELS for c in folded)
    if n:
        return n
    n = sum(
        1 for c in text
        if (0x3040 <= ord(c) <= 0x30FF)   # Hiragana/Katakana
        or (0x3400 <= ord(c) <= 0x9FFF)   # CJK ideographs
        or (0xAC00 <= ord(c) <= 0xD7A3)   # Hangul syllables
    )
    if n:
        return n
    return max(1, len(text.replace(" ", "")))


def _ensure_gguf(model_path: Optional[str]) -> str:
    """Resolve the GGUF path from MODEL_WEIGHTS_DIR; download on first use."""
    if model_path and os.path.exists(model_path):
        return model_path
    local_dir = os.path.join(MODEL_WEIGHTS_DIR, _GGUF_DIR)
    candidate = os.path.join(local_dir, _GGUF_FILE)
    if os.path.exists(candidate):
        return candidate
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=_GGUF_REPO,
        filename=_GGUF_FILE,
        local_dir=local_dir,
        token=os.environ.get("HF_TOKEN"),
    )


class LlamaCppTranslationClient:
    """Gemma-4-E2B translation via llama-cpp-python, with an isochrony loop."""

    def __init__(self, device: str = "cuda", model_path: Optional[str] = None):
        self.device = device
        self.model_path = _ensure_gguf(model_path)
        self._llm = None

    def load_models(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama, LLAMA_SPLIT_MODE_NONE
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `llama-cpp-python` required for LlamaCppTranslationClient. "
                "Install with: pip install llama-cpp-python"
            ) from e

        # -1 offloads the whole model to GPU on cuda; 0 keeps it on CPU otherwise.
        # Only the text GGUF is loaded (no mmproj) so the vision encoder is never
        # instantiated.
        n_gpu_layers = -1 if self.device.startswith("cuda") else 0
        n_threads = os.cpu_count() or 8

        logger.info("Loading gemma-4-E2B GGUF via llama.cpp: %s (n_gpu_layers=%s)",
                    self.model_path, n_gpu_layers)
        # split_mode=NONE + main_gpu=0 pins llama.cpp to a single GPU. On a
        # multi-GPU host GGML would otherwise register one backend per device and
        # Gemma-4's SWA graph trips GGML_SCHED_MAX_SPLIT_INPUTS when split across
        # them; pinning makes it behave like the single-GPU case.
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            flash_attn=True,
            split_mode=LLAMA_SPLIT_MODE_NONE,
            main_gpu=0,
            verbose=False,
        )

    def _generate(self, user_content: str) -> str:
        """One no-think translation turn (default template => empty thought channel)."""
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    def _translate_one(self, text: str, src: str, tgt: str) -> str:
        system = (
            f"You are a professional dubbing translator. Translate the line from "
            f"{src} to {tgt}. Output ONLY the {tgt} translation — no notes, "
            f"explanations, quotes, or alternatives. Preserve the speaker's tone, "
            f"register, and emotion. Ensure the translation sounds natural and "
            f"idiomatic in {tgt} rather than a literal word-for-word rendering. "
            f"Keep the spoken length close to the original for dubbing sync."
        )
        target_len = _spoken_length(text)

        out = self._generate(f"{system}\n\nLine: {text}")
        # Track every attempt (incl. the clean single-shot) and ultimately return
        # the one whose spoken length is closest to the source — this is the guard
        # against length pressure degrading meaning: a worse-length retry is dropped.
        best, best_err = out, abs(_spoken_length(out) / target_len - 1.0)

        for _ in range(_ISO_MAX_ITERS):
            cur_len = _spoken_length(out)
            ratio = cur_len / target_len
            if 1 - _ISO_TOL <= ratio <= 1 + _ISO_TOL:
                return out
            longer = ratio < 1
            # Flat re-prompt: only the LATEST attempt is carried into context, never
            # the whole iteration history, so the prompt stays small no matter how
            # many times we loop.
            feedback = (
                f"{system}\n\nLine: {text}\n\n"
                f'Your previous {tgt} translation was: "{out}"\n'
                f"It is {'too short' if longer else 'too long'} for dubbing "
                f"(needs about {target_len} syllables, it has {cur_len}). Give a "
                f"{'longer' if longer else 'shorter'} {tgt} translation with the "
                f"same meaning. Output ONLY the translation."
            )
            out = self._generate(feedback)
            err = abs(_spoken_length(out) / target_len - 1.0)
            if err < best_err:
                best, best_err = out, err

        return best

    def translate(
        self,
        sentences: List[str],
        source_language: str,
        target_language: str,
    ) -> List[str]:
        if self._llm is None:
            raise RuntimeError("Call load_models() before translate().")

        src = _lang_name(source_language)
        tgt = _lang_name(target_language)

        results: List[str] = []
        for sentence in sentences:
            text = (sentence or "").strip()
            if not text:
                results.append("")
                continue
            results.append(self._translate_one(text, src, tgt))
        logger.info("llama.cpp translated %d sentences (%s → %s)", len(sentences), src, tgt)
        return results
