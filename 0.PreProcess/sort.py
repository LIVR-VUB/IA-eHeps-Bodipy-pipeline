#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

# === Edit these two paths ===
SRC_ROOT = Path("/home/arka/Desktop/Yuwei/Raw/Hep_bodipy__cellmask_test_20250821")
DEST_DIR = Path("/home/arka/Desktop/Yuwei/Raw/sorted")

# Set to True to MOVE files (remove from source). False will COPY.
DO_MOVE = False

# --- You probably don't need to change anything below ---
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Match files that:
#   1) have "_R_" somewhere in the filename
#   2) end with d0 / d1 / d3 before the extension (case-insensitive .tif/.tiff)
# Examples: ...A03f00d0.TIF, ...B02f05d3.tif
pattern = re.compile(r"_R_.*d([013])\.tif{1,2}$", re.IGNORECASE)

selected = []
for p in SRC_ROOT.rglob("*"):
    if p.is_file() and pattern.search(p.name):
        selected.append(p)

print(f"Found {len(selected)} matching files.")

def safe_dest_path(dest_dir: Path, filename: str) -> Path:
    """Avoid overwriting by appending -1, -2, ... if needed."""
    out = dest_dir / filename
    if not out.exists():
        return out
    stem = out.stem
    suffix = out.suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

moved = 0
copied = 0
for src in selected:
    dest = safe_dest_path(DEST_DIR, src.name)
    if DO_MOVE:
        shutil.move(str(src), dest)
        moved += 1
    else:
        shutil.copy2(src, dest)
        copied += 1

op = "moved" if DO_MOVE else "copied"
print(f"Done: {moved if DO_MOVE else copied} files {op} to {DEST_DIR}")
