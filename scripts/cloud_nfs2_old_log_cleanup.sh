#!/usr/bin/env bash
set -u

root=/workspace-SR006.nfs2
logs="$root/mlsub-logs"
: "${ACTIVE_JOB_IDS:?set ACTIVE_JOB_IDS to a colon-separated global active-job list}"

echo "=== before ==="
df -h "$root"
du -sh "$logs" 2>/dev/null || true

removed=0
removed_kib=0
while IFS= read -r -d '' path; do
  name=${path##*/}
  case ":$ACTIVE_JOB_IDS:" in
    *":$name:"*)
      echo "KEEP_ACTIVE $name"
      continue
      ;;
  esac
  size=$(du -sk "$path" 2>/dev/null | awk '{print $1}')
  size=${size:-0}
  rm -rf -- "$path"
  echo "REMOVED_OLD $name ${size}KiB"
  removed=$((removed + 1))
  removed_kib=$((removed_kib + size))
done < <(find "$logs" -mindepth 1 -maxdepth 1 -type d -name 'lm-mpi-job-*' -mtime +7 -print0)

echo "REMOVED_COUNT=$removed REMOVED_KIB=$removed_kib"
echo "=== after ==="
df -h "$root"
du -sh "$logs" 2>/dev/null || true
echo "EXIT=0"
