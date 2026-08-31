#!/usr/bin/env bash
set -euo pipefail

echo "=== filesystems ==="
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1
df -i /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1

echo "=== configured parquet root ==="
datasets_dir=${DATASETS_DIR:-/home/jovyan/data/fineweb-edu/sample/100BT}
if [[ -d $datasets_dir ]]; then
  find "$datasets_dir" -maxdepth 1 -type f -name '*.parquet' -printf '%s %p\n' | awk '{bytes += $1; count += 1} END {print "parquet_files=" count, "parquet_bytes=" bytes}'
else
  echo "missing: $datasets_dir"
fi

echo "=== other parquet candidates ==="
for root in /home/jovyan/data /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
  [[ -d $root ]] || continue
  find "$root" -maxdepth 8 -type f -name '*.parquet' -print -quit 2>/dev/null || true
done
