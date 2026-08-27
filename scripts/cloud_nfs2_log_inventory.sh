#!/usr/bin/env bash
set -u

root=/workspace-SR006.nfs2
logs="$root/mlsub-logs"

echo "=== filesystem ==="
df -h "$root"
df -i "$root"

echo "=== nfs2 top level ==="
du -sh "$root"/* 2>/dev/null | sort -h

echo "=== mlsub log directories by size (KiB) ==="
if [[ -d $logs ]]; then
  find "$logs" -mindepth 1 -maxdepth 1 -type d -print0 \
    | xargs -0 -r du -sk 2>/dev/null \
    | sort -nr \
    | head -100
  echo "=== mlsub log directories by mtime ==="
  find "$logs" -mindepth 1 -maxdepth 1 -type d \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %p\n' \
    | sort \
    | head -100
else
  echo "$logs is absent"
fi

echo "EXIT=0"
