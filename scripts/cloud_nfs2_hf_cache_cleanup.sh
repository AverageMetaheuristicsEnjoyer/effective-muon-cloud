#!/usr/bin/env bash
set -u

root=/workspace-SR006.nfs2
target="$root/hmoe-hf-cache"

echo "=== before ==="
df -h "$root"
du -sh "$target" 2>/dev/null || true

if [[ -d $target ]]; then
  rm -rf -- "$target"
  echo "REMOVED $target"
else
  echo "SKIP $target: absent"
fi

echo "=== after ==="
df -h "$root"
echo "EXIT=0"
