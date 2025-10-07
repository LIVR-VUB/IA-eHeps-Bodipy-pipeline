#!/usr/bin/env python3
"""
Apply a trained CAREamics Noise2Void model to TIFF images (any size) using tiling.

Example:
  python apply_n2v.py \
    --model /home/arka/Desktop/Milan/model/n2v_liver_model.zip \
    --input /home/arka/Desktop/Milan/Raw/Green_raw/Control \
    --output /home/arka/Desktop/Milan/denoised \
    --glob "*tif" --z 0 --tile 64 64 --overlap 32 32 --no_tta
"""

import argparse
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from careamics import CAREamist

# Optional: RTX Tensor Cores
try:
    import torch
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _normalize01(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Return float32 image in [0,1] and a scale to recover original dtype."""
    img = img.astype(np.float32, copy=False)
    if img.dtype == np.uint8:
        return img / 255.0, 255.0
    if img.dtype == np.uint16:
        return img / 65535.0, 65535.0
    m = float(img.max())
    return (img / m if m > 0 else img), (m if m > 1.0 else 1.0)


def _prepare_2d(img: np.ndarray, take_z: int | None) -> np.ndarray:
    """Reduce any TIFF to a single 2D plane (Y,X)."""
    if img.ndim == 4:
        # Heuristic: (C,Z,Y,X) vs (Z,C,Y,X)
        if img.shape[0] <= 4 and img.shape[1] > 4:  # (C,Z,Y,X)
            img = img[0]
            img = img[take_z] if (take_z is not None and img.shape[0] > 1) else img.max(axis=0)
        else:  # (Z,C,Y,X)
            img = img[:, 0]
            img = img[take_z] if (take_z is not None and img.shape[0] > 1) else img.max(axis=0)
    elif img.ndim == 3:  # (Z,Y,X) or (C,Y,X)
        if take_z is not None and img.shape[0] > 1:
            img = img[min(max(int(take_z), 0), img.shape[0] - 1)]
        else:
            img = img if img.shape[0] == 1 else img.max(axis=0)
    elif img.ndim != 2:
        raise ValueError(f"Unsupported image shape {img.shape}")
    return np.squeeze(img)


def _load_model(path: str) -> CAREamist:
    """Load BMZ .zip, Lightning .ckpt, or a model folder via the main constructor."""
    p = Path(path)
    try:
        return CAREamist(str(p))
    except Exception:
        pass
    if hasattr(CAREamist, "from_folder") and p.is_dir():
        return CAREamist.from_folder(str(p))
    raise RuntimeError(f"Could not load model from: {p}")


def denoise_one(
    model: CAREamist,
    in_path: Path,
    out_path: Path,
    take_z: int | None,
    tile: tuple[int, int],
    overlap: tuple[int, int],
    tta: bool,
    batch_size: int,
):
    x = imread(str(in_path))
    x2d = _prepare_2d(x, take_z=take_z)   # (Y, X)
    x01, scale = _normalize01(x2d)        # float32 in [0,1]

    # Pass what we provide: 2D => axes="YX"
    y_pred = model.predict(
        source=x01,
        axes="YX",
        data_type="array",
        tile_size=list(tile),        # e.g. [64, 64]
        tile_overlap=list(overlap),  # e.g. [32, 32] (even)
        tta=tta,
        batch_size=batch_size,
    )

    y_pred = np.clip(y_pred, 0.0, 1.0)
    if scale == 255.0:
        y_out = (y_pred * 255.0 + 0.5).astype(np.uint8)
    elif scale == 65535.0:
        y_out = (y_pred * 65535.0 + 0.5).astype(np.uint16)
    else:
        y_out = y_pred.astype(np.float32)

    imwrite(str(out_path), y_out, photometric="minisblack")


def main():
    ap = argparse.ArgumentParser(description="CAREamics N2V tiled inference.")
    ap.add_argument("--model", required=True, help="Path to model (.zip/.ckpt/folder)")
    ap.add_argument("--input", required=True, help="Folder of input TIFFs")
    ap.add_argument("--output", required=True, help="Folder to write denoised TIFFs")
    ap.add_argument("--glob", default="*.tif", help="Pattern for input files")
    ap.add_argument("--z", type=int, default=None, help="Z index to take if stack (default: max projection)")
    ap.add_argument("--tile", nargs=2, type=int, default=[64, 64], metavar=("H", "W"),
                    help="Tile size (use training patch size; multiple of 2^n recommended)")
    ap.add_argument("--overlap", nargs=2, type=int, default=[32, 32], metavar=("H", "W"),
                    help="Even overlap per tile (e.g., 32)")
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

    for f in files:
        # Print original shape just for visibility
        try:
            shp = imread(str(f)).shape
        except Exception:
            shp = "?"
        print(f"[DBG] {f.name}: using axes=YX, tile={tuple(args.tile)}, overlap={tuple(args.overlap)}, tta={args.tta}, in_shape={shp}")

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
    print("[DONE]")


if __name__ == "__main__":
    main()
