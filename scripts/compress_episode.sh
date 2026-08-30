#!/usr/bin/env bash
# Shrink a rendered episode for in-chat delivery.
#
#   bash scripts/compress_episode.sh video/topics/p0.1/out/p0-1-e1.mp4 [outdir]
#
# Remotion's default H.264 output is ~3.7 MB per minute, which puts a
# fifteen-minute episode over the 30 MiB chat upload limit. These slides are
# static for seconds at a time, so x264 at CRF 23 reproduces them essentially
# losslessly at roughly a third of the size, and mono 64 kbps AAC is plenty for
# a single narrator at 24 kHz.
#
# The master render stays untouched; this writes a separate delivery copy.
set -euo pipefail

SRC="${1:?usage: compress_episode.sh <input.mp4> [outdir]}"
OUTDIR="${2:-$(dirname "$SRC")}"
NAME="$(basename "${SRC%.mp4}")"
DEST="$OUTDIR/${NAME}-delivery.mp4"

mkdir -p "$OUTDIR"
ffmpeg -y -v error -i "$SRC" \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 64k -ac 1 \
  "$DEST"

before=$(stat -c%s "$SRC")
after=$(stat -c%s "$DEST")
printf '%s\n  %.1f MiB -> %.1f MiB (%.0f%%)\n' "$DEST" \
  "$(echo "$before/1048576" | bc -l)" "$(echo "$after/1048576" | bc -l)" \
  "$(echo "100*$after/$before" | bc -l)"
