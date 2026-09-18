import base64
import gzip
import hashlib
import re
import sys
from pathlib import Path


destination = Path(sys.argv[1])
chunks = {}
for line in sys.stdin:
    line = line.replace("[1,0]<stdout>:", "")
    match = re.search(r"RESULT_CHUNK (\S+) (\d+) (\d+) ([0-9a-f]{64}) ([A-Za-z0-9+/=]+)", line)
    if match:
        name, index, total, digest, data = match.groups()
        if Path(name).name != name:
            raise ValueError(name)
        entry = chunks.setdefault(name, dict(total=int(total), sha256=digest, parts={}))
        if entry["total"] != int(total) or entry["sha256"] != digest:
            raise ValueError(f"Conflicting export: {name}")
        entry["parts"][int(index)] = data
if not chunks:
    raise RuntimeError("No exported artifacts found")
decoded = {}
for name, entry in chunks.items():
    if set(entry["parts"]) != set(range(entry["total"])):
        raise RuntimeError(f"Incomplete export: {name}")
    raw = gzip.decompress(base64.b64decode("".join(entry["parts"][i] for i in range(entry["total"]))))
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {name}")
    path = destination / name
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError(path)
    decoded[name] = raw
destination.mkdir(parents=True, exist_ok=True)
for name, raw in decoded.items():
    (destination / name).write_bytes(raw)
    print("VERIFIED", name, hashlib.sha256(raw).hexdigest())
print("VERIFIED_FILES", len(decoded))
