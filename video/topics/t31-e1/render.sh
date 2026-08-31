#!/usr/bin/env bash
# Render t31-e1 at 1280x720 / 30 fps.
#
#   bash video/topics/t31-e1/render.sh            # -> out/t31-e1.mp4
#
# The mp4 is gitignored; it is regenerable from script.md + timings.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/out/t31-e1.mp4"

mkdir -p "$HERE/out"
if [ ! -f "$HERE/timings.json" ]; then
  echo "no timings.json — run: python3 video/kokoro/narrate.py video/topics/t31-e1/script.md" >&2
  exit 1
fi

cd "$VIDEO"
npx remotion render src/index.ts t31-e1 "$OUT" \
  --codec=h264 --image-format=jpeg --concurrency=2 --log=info

echo "wrote $OUT"
