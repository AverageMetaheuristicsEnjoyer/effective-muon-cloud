import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


source = Path("/workspace-SR006.nfs3/layerwise-tucker-riemann-20260917")
destination = Path(sys.argv[1])
if destination.resolve() == source.resolve() or source.resolve() in destination.resolve().parents:
    raise ValueError("Recovery requires a new root on another disk")
destination.mkdir(parents=True, exist_ok=False)
hashes = {}
for group in ("small", "large"):
    old = source / f"scale-{group}"
    new = destination / f"original-{group}"
    paths = sorted(old.glob("*.json")) + sorted(old.glob("*.exit"))
    paths += sorted((old / "riemann-scaling").glob("*.json"))
    paths += sorted((old / "logs").glob("*.log"))
    counts = Counter()
    for path in paths:
        target = new / path.relative_to(old)
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = path.read_bytes()
        target.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Backup checksum mismatch: {target}")
        hashes[str(target.relative_to(destination))] = digest
        if path.parent.name == "riemann-scaling" and not path.name.startswith("manifest-"):
            try:
                counts[json.loads(raw)["status"]] += 1
            except (json.JSONDecodeError, KeyError):
                counts["invalid"] += 1
    print("BACKUP", group, json.dumps(counts), flush=True)
(destination / "original-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
print("VERIFIED_BACKUP_FILES", len(hashes), flush=True)
for group in ("small", "large"):
    for name in ("inductor-cache", "triton-cache"):
        cache = source / f"scale-{group}" / name
        if cache.exists():
            if cache.is_symlink():
                raise ValueError(f"Refusing to remove a symlink: {cache}")
            shutil.rmtree(cache)
            print("REMOVED_REGENERABLE_CACHE", cache, flush=True)
print("RECOVERY_EXIT=0", flush=True)
