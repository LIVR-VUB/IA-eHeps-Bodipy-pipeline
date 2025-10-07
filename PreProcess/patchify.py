#!/usr/bin/env python3
"""
Patchify TIFF images into overlapping tiles for self-supervised training (Noise2Void).

Usage:
  python patchify_careamics.py \
      --input /path/to/raw_tifs \
      --output /path/to/patches \
      --patch_size 64 \
      --stride 32 \
      --glob "*.TIF"
"""

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
from tifffile import imread, imwrite
from skimage.util import view_as_windows


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max to [0,255] uint8 (per image)."""
    arr = arr.astype(np.float32)
    a, b = np.min(arr), np.max(arr)
    if b > a:
        arr = (arr - a) / (b - a)
    else:
        arr = np.zeros_like(arr)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def patchify_image(
    img_path: Path,
    out_dir: Path,
    patch_size: int,
    stride: int,
    take_z: int | None = 0,
    channel: int | None = None,
) -> int:
    """
    Extract patches from a 2D image. If Z-stack, take Z 'take_z' (default 0).
    If multi-channel, pick 'channel' or use single-channel if None.
    Returns number of patches written.
    """
    img = imread(str(img_path), is_ome=False)

    # Reduce Z if 3D (Z,Y,X) or (C,Z,Y,X)
    if img.ndim == 4:
        # Try (C,Z,Y,X) first
        if channel is not None:
            img = img[channel]
        else:
            img = img[0]
        img = img[take_z] if take_z is not None else img.max(axis=0)
    elif img.ndim == 3:
        # Could be (Z,Y,X) or (C,Y,X); assume Z,Y,X if take_z provided
        if take_z is not None and img.shape[0] > 1:
            img = img[take_z]
        else:
            img = img if img.shape[0] == 1 else img.max(axis=0)
    elif img.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported ndim={img.ndim} for {img_path}")

    # Convert to uint8 (Noise2Void examples typically use [0,1]/uint8)
    if img.dtype != np.uint8:
        img_u8 = _normalize_to_uint8(img)
    else:
        img_u8 = img

    H, W = img_u8.shape
    if H < patch_size or W < patch_size:
        print(f"[SKIP] {img_path.name} too small: {img_u8.shape}")
        return 0

    windows = view_as_windows(img_u8, (patch_size, patch_size), step=stride)
    base = img_path.stem
    count = 0
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            patch = np.ascontiguousarray(windows[i, j])
            out_path = out_dir / f"{base}_p{patch_size}_s{stride}_{count:05d}.tif"
            imwrite(out_path, patch, photometric="minisblack")
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="Patchify images for N2V training.")
    ap.add_argument("--input", required=True, help="Folder with input TIFFs")
    ap.add_argument("--output", required=True, help="Folder to write patches")
    ap.add_argument("--patch_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--glob", default="*.tif", help="Pattern for input files")
    ap.add_argument("--z", type=int, default=0, help="Z index to take if stack")
    ap.add_argument("--channel", type=int, default=None, help="Channel index if multi-channel")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    total = 0
    for f in files:
        n = patchify_image(f, out_dir, args.patch_size, args.stride, take_z=args.z, channel=args.channel)
        print(f"[OK] {f.name}: {n} patches")
        total += n
    print(f"[DONE] Total patches: {total}")


if __name__ == "__main__":
    main()
