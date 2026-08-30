#!/usr/bin/env bash
# Fetch the Kokoro-82M ONNX weights (Apache-2.0). ~350 MB, not committed.
#
#   bash video/kokoro/fetch_model.sh
#
# Override the destination with FORGE_KOKORO_DIR.
set -euo pipefail

DIR="${FORGE_KOKORO_DIR:-$HOME/.cache/forge/kokoro}"
BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

mkdir -p "$DIR"
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  if [ -s "$DIR/$f" ]; then
    echo "have $DIR/$f"
  else
    echo "fetching $f -> $DIR"
    curl -sSL -o "$DIR/$f" "$BASE/$f"
  fi
done
echo "done. python3 -c \"import kokoro_onnx\" to check the runtime is installed."
