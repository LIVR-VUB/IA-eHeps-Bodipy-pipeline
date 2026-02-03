#!/usr/bin/env python3
"""
Apply a trained CAREamics Noise2Void model to TIFF images (any size) using tiling.

Example:
  python apply_n2v.py \
    --model /path/to/n2v_model.zip \
    --input /path/to/input_tiffs \
    --output /path/to/denoised_out \
    --glob "*.[Tt][Ii][Ff]*" \
    --tile 256 256 --overlap 48 48 --no_tta
"""

import argparse
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from careamics import CAREamist

# Optional: faster matmul on modern GPUs (safe no-op on CPU)
try:
    import torch
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _normalize01(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Return float32 image in [0,1] and a scale to recover original dtype range."""
    orig_dtype = img.dtype
    if orig_dtype == np.uint8:
        return img.astype(np.float32) / 255.0, 255.0
    if orig_dtype == np.uint16:
        return img.astype(np.float32) / 65535.0, 65535.0
    img = img.astype(np.float32)
    m = float(img.max())
    return (img / m if m > 0 else img), (m if m > 1.0 else 1.0)


def _prepare_2d(img: np.ndarray, take_z: int | None) -> np.ndarray:
    """
    Reduce any TIFF to a single 2D plane (Y,X).
    - (Y,X)              -> as is
    - (Z,Y,X) or (C,Y,X) -> take Z index if provided, else max-projection
    - (C,Z,Y,X)          -> use C=0, then Z as above
    - (Z,C,Y,X)          -> use C=0, then Z as above
    """
    if img.ndim == 2:
        return img

    if img.ndim == 3:
        # (Z,Y,X) or (C,Y,X)
        n = img.shape[0]
        if take_z is not None and n > 1:
            return img[min(max(int(take_z), 0), n - 1)]
        return img if n == 1 else img.max(axis=0)

    if img.ndim == 4:
        # Heuristic for (C,Z,Y,X) vs (Z,C,Y,X)
        if img.shape[0] <= 4 and img.shape[1] > 4:  # (C,Z,Y,X)
            img = img[0]  # C=0
        else:  # (Z,C,Y,X)
            img = img[:, 0]  # C=0
        n = img.shape[0]
        if take_z is not None and n > 1:
            return img[min(max(int(take_z), 0), n - 1)]
        return img if n == 1 else img.max(axis=0)

    raise ValueError(f"Unsupported image shape {img.shape}")


def _load_model(path: str) -> CAREamist:
    """Load BMZ/ZIP, CKPT, or a model folder via CAREamist constructor."""
    p = Path(path)
    try:
        return CAREamist(str(p))
    except Exception:
        pass
    if hasattr(CAREamist, "from_folder") and p.is_dir():
        return CAREamist.from_folder(str(p))
    raise RuntimeError(f"Could not load model from: {p}")


def _predict_2d(model: CAREamist, x01: np.ndarray,
                tile: tuple[int, int], overlap: tuple[int, int],
                tta: bool, batch_size: int) -> np.ndarray:
    """
    Run CAREamics predict on a single 2D float image in [0,1].
    Handles both ndarray and list-like returns across versions.
    """
    res = model.predict(
        source=x01,                 # (Y,X) float32
        axes="YX",
        data_type="array",
        tile_size=list(tile),
        tile_overlap=list(overlap),
        tta=tta,
        batch_size=batch_size,
    )
    if isinstance(res, np.ndarray):
        return res
    if isinstance(res, (list, tuple)) and len(res) > 0:
        r0 = res[0]
        return r0[0] if isinstance(r0, (list, tuple)) and len(r0) > 0 else r0
    raise RuntimeError("Unexpected predict() return type.")


def denoise_one(model: CAREamist, in_path: Path, out_path: Path,
                take_z: int | None, tile: tuple[int, int],
                overlap: tuple[int, int], tta: bool, batch_size: int):
    x = imread(str(in_path))
    x2d = _prepare_2d(x, take_z=take_z)           # (Y, X)
    x2d = np.squeeze(x2d)                          # ensure 2D input
    if x2d.ndim != 2:
        raise RuntimeError(f"Input not 2D after squeeze: {x2d.shape}")

    x01, scale = _normalize01(x2d)                 # float32 in [0,1]
    y = _predict_2d(model, x01, tile, overlap, tta, batch_size)

    # 🔧 ensure strictly 2D for Cellpose
    y = np.asarray(y)
    y = np.squeeze(y)                              # drop (S, C) singleton axes if present
    if y.ndim != 2:
        raise RuntimeError(f"Predicted array not 2D after squeeze: shape={y.shape}")

    y = np.clip(y, 0.0, 1.0)

    # restore dtype / dynamic range
    if scale == 255.0:
        y = (y * 255.0 + 0.5).astype(np.uint8)
    elif scale == 65535.0:
        y = (y * 65535.0 + 0.5).astype(np.uint16)
    else:
        y = y.astype(np.float32)

    imwrite(str(out_path), y, photometric="minisblack")


def main():
    ap = argparse.ArgumentParser(description="CAREamics Noise2Void tiled inference.")
    ap.add_argument("--model", required=True, help="Path to model (.zip/.bmz/.ckpt or folder)")
    ap.add_argument("--input", required=True, help="Folder of input TIFFs")
    ap.add_argument("--output", required=True, help="Folder to write denoised TIFFs")
    ap.add_argument("--glob", default="*.[Tt][Ii][Ff]*", help="Pattern for input files")
    ap.add_argument("--z", type=int, default=None, help="Z index if stack (default: max projection)")
    ap.add_argument("--tile", nargs=2, type=int, default=[256, 256], metavar=("H", "W"),
                    help="Tile size (bigger is faster; 256x256 is a good default)")
    ap.add_argument("--overlap", nargs=2, type=int, default=[48, 48], metavar=("H", "W"),
                    help="Tile overlap to reduce seams (10–25% of tile)")
    ap.add_argument("--batch_size", type=int, default=1, help="Tiles per step")
    ap.add_argument("--tta", dest="tta", action="store_true", help="Enable test-time augmentation")
    ap.add_argument("--no_tta", dest="tta", action="store_false")
    ap.set_defaults(tta=False)
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.model)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files match {args.glob} in {in_dir}")

    # sanity: overlap must be smaller than tile
    ty, tx = args.tile
    oy, ox = args.overlap
    if oy >= ty or ox >= tx:
        raise ValueError(f"Overlap {args.overlap} must be < tile {args.tile}")

    for f in files:
        try:
            shp = imread(str(f)).shape
        except Exception:
            shp = "?"
        print(f"[DBG] {f.name}: axes=YX, tile={tuple(args.tile)}, overlap={tuple(args.overlap)}, "
              f"tta={args.tta}, in_shape={shp}")

        out = out_dir / f"{f.stem}_denoised.tif"
        denoise_one(
            model=model,
            in_path=f,
            out_path=out,
            take_z=args.z,
            tile=tuple(args.tile),
            overlap=tuple(args.overlap),
            tta=args.tta,
            batch_size=args.batch_size,
        )
        print(f"[OK]  {f.name} -> {out.name}")
    print("[DONE] All files written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
