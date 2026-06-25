#!/usr/bin/env python3
"""Generate a small synthetic test video with a talking-head-style pattern.

No external downloads needed — uses ffmpeg (must be on PATH) to create
a short test clip suitable for running the pipeline end to end.

Usage:
    python tests/fixtures/generate_test_video.py
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "media"
OUTPUT_VIDEO = OUTPUT_DIR / "test_video.mp4"
DURATION = 5  # seconds


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_VIDEO.exists():
        print(f"  ✓ {OUTPUT_VIDEO.name} already exists ({OUTPUT_VIDEO.stat().st_size // 1024} KB)")
        return

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={DURATION}:size=480x270:rate=24",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={DURATION}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        str(OUTPUT_VIDEO),
    ]

    print(f"  Generating {OUTPUT_VIDEO.name} ({DURATION}s, 480x270) ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print(f"  ❌ ffmpeg failed:\n{result.stderr[:300]}")
        sys.exit(1)

    size_kb = OUTPUT_VIDEO.stat().st_size // 1024
    print(f"  ✅ {OUTPUT_VIDEO.name} ({size_kb} KB)")


if __name__ == "__main__":
    main()
