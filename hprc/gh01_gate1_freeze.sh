#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
TAG=${TAG:-gh01_20260724}
DATA=${DATA:-$ROOT/data/500um_p512_n500}
REPORT_ROOT="$ROOT/reports/$TAG"
LOG_ROOT="$ROOT/logs/$TAG"
RUN_ROOT="$ROOT/runs/$TAG"

mkdir -p "$REPORT_ROOT" "$LOG_ROOT" "$RUN_ROOT"

OUT="$REPORT_ROOT/gate1_freeze.txt"
TMP_OUT="$OUT.tmp"
MANIFEST="$REPORT_ROOT/data_manifest.sha256.csv"

exec > >(tee "$TMP_OUT") 2>&1

echo "[GATE1] start $(date -Is)"
echo "[GATE1] root=$ROOT tag=$TAG data=$DATA"
echo

test -d "$ROOT"
test -d "$ROOT/src"
test -d "$DATA"

echo "===== git ====="
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ROOT" rev-parse HEAD || true
  git -C "$ROOT" status --short || true
  git -C "$ROOT" diff --stat || true
else
  echo "not a git worktree on ACES project root"
fi
echo

echo "===== python env ====="
PY_FILE="$REPORT_ROOT/python_cuda_ok.txt"
if [[ -f "$PY_FILE" ]]; then
  PY=$(head -n 1 "$PY_FILE")
elif [[ -n "${PY:-}" ]]; then
  PY="$PY"
else
  PY=$(command -v python3 || true)
fi
echo "PY=$PY"
if [[ -n "$PY" && -x "$PY" ]]; then
  "$PY" - <<'PY' || true
import json, platform, sys
out = {"executable": sys.executable, "python": sys.version, "machine": platform.machine()}
for name in ["torch", "diffusers", "numpy", "PIL"]:
    try:
        mod = __import__(name)
        out[name] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        out[name + "_error"] = repr(exc)
try:
    import torch
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
except Exception as exc:
    out["torch_probe_error"] = repr(exc)
print(json.dumps(out, indent=2, sort_keys=True))
PY
fi
echo

echo "===== data manifest ====="
tmp_manifest="$MANIFEST.tmp"
rm -f "$tmp_manifest"
printf 'relative_path,size_bytes,sha256\n' > "$tmp_manifest"
while IFS= read -r -d '' rel; do
  path="$DATA/$rel"
  size=$(stat -c '%s' "$path")
  hash=$(sha256sum "$path" | awk '{print $1}')
  printf '%s,%s,%s\n' "$rel" "$size" "$hash" >> "$tmp_manifest"
done < <(
  cd "$DATA"
  find . -maxdepth 1 -type f \( \
    -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o \
    -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \
  \) -printf '%P\0' | sort -z
)
mv "$tmp_manifest" "$MANIFEST"
count=$(($(wc -l < "$MANIFEST") - 1))
echo "image_count=$count"
echo "manifest=$MANIFEST"
sha256sum "$MANIFEST" | tee "$REPORT_ROOT/data_manifest.file_sha256.txt"
if [[ "$count" -ne 4000 ]]; then
  echo "[GATE1] FAILED: expected 4000 images, found $count"
  mv "$TMP_OUT" "$OUT"
  exit 4
fi
echo

echo "===== source summary ====="
"$PY" - <<'PY' "$MANIFEST" || true
import csv, re, sys
from collections import Counter

manifest = sys.argv[1]
source_counts = Counter()
process_counts = Counter()
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        stem = row["relative_path"].rsplit(".", 1)[0]
        source = stem.split("_p", 1)[0]
        source_counts[source] += 1
        process_counts[source.split(".", 1)[0]] += 1
print("sources")
for key, value in sorted(source_counts.items()):
    print(f"{key}: {value}")
print("processes")
for key, value in sorted(process_counts.items()):
    print(f"{key}: {value}")
PY

echo "[GATE1] done $(date -Is)"
mv "$TMP_OUT" "$OUT"
