# Test Fixtures

This directory holds input data for the test suite.

## `media/` — Example video files for pipeline testing

| File | Source | Description |
|------|--------|-------------|
| `test_video.mp4` | ffmpeg (synthetic) | 5s color bars + 440 Hz tone, 480×270. Good for local pipeline smoke tests. |
| `1.mp4` | S3 ([download](#downloading-from-s3)) | ~30s Russian speech clip. Used by `test_input.json`. |

### Generating locally

```bash
python tests/fixtures/generate_test_video.py
```

Requires `ffmpeg` on `PATH`. Creates `media/test_video.mp4` (~120 KB).

### Downloading from S3

```bash
bash tests/fixtures/download_test_media.sh
```

Downloads example clips from the project's public Yandex Cloud bucket
to `media/`. Requires `curl`.
