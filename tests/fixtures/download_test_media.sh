#!/usr/bin/env bash
set -euo pipefail

# Downloads example test media for the langswap pipeline.
# These are short clips hosted in the project's public S3 bucket.
# Alternatively, run `python generate_test_video.py` for a synthetic video
# that requires no download.

BASE_URL="https://storage.yandexcloud.net/langswap-public"
DEST="$(cd "$(dirname "$0")" && pwd)/media"

mkdir -p "$DEST"

echo "Downloading test media to $DEST ..."

for url in \
    "$BASE_URL/ru_source/1.mp4" \
    "$BASE_URL/en_source/sample_en.mp4"; do
    filename="${url##*/}"
    if [ -f "$DEST/$filename" ]; then
        echo "  ✓ $filename already exists"
    else
        echo "  ↓ $filename ..."
        curl -fL "$url" -o "$DEST/$filename"
        echo "  ✓ $filename"
    fi
done

echo "Done. Test media is ready."
