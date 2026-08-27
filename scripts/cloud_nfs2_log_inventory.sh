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
  du -ah "$logs" 2>/dev/null | sort -rh | head -100
  echo "=== mlsub log entries by mtime ==="
  find "$logs" -mindepth 1 -maxdepth 2 \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %y %s %p\n' \
    | sort \
    | head -100
else
  echo "$logs is absent"
fi

echo "EXIT=0"
