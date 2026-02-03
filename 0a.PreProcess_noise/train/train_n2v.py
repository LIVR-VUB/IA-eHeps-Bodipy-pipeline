#!/usr/bin/env python3
r"""
Train a Noise2Void (N2V) model with CAREamics using TIFF patches.
- Cross-platform
- Writes logs/ckpts into --save_dir
- Prints best checkpoint
- Optional BMZ export (fixed: input_array matches axes)

Run command:

python train_n2v.py \
  --patch_dir /path/to/patches \
  --save_dir /path/to/save_dir \
  --epochs 50 \
  --batch_size 16 \
  --y_patch 64 \
  --x_patch 64 \
  --axes auto \
  --use_augmentations \
  --num_workers 8 \
  --export_bmz \
  --bmz_name n2v_liver_model \
  --bmz_desc "Noise2Void model for GFP liver slice denoising"

"""

import argparse, os, platform
from pathlib import Path
from glob import glob
import numpy as np
from tifffile import imread
from careamics import CAREamist
from careamics.config import create_n2v_configuration

try:
    import torch
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def load_patches_stack(patch_dir: Path) -> np.ndarray:
    paths = sorted(glob(str(patch_dir / "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No TIFF patches found in: {patch_dir}")
    imgs = []
    for p in paths:
        x = imread(p)
        x = np.squeeze(x)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D patch, got {x.shape} for {p}")
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        else:
            x = x.astype(np.float32)
            m = float(x.max())
            if m > 0:
                x /= (m if m > 1.0 else 1.0)
        imgs.append(x)
    return np.stack(imgs, axis=0)  # (N, Y, X)


def decide_axes_and_patchsize(data_ndim: int, args) -> tuple[str, tuple]:
    req = args.axes.upper()
    if req == "AUTO":
        if data_ndim == 3:
            axes = "SYX"
        elif data_ndim == 2:
            axes = "YX"
        elif data_ndim == 4:
            axes = "SZYX"
        else:
            raise ValueError(f"Unsupported data ndim {data_ndim} for AUTO axes.")
    else:
        axes = req

    if axes in ("YX", "SYX"):
        patch_size = (int(args.y_patch), int(args.x_patch))
    elif axes in ("ZYX", "SZYX"):
        z = int(args.z_patch or 1)
        if z < 9:
            print(f"[INFO] z_patch={z} too small for 3D; switching to 2D 'SYX'.")
            axes = "SYX"
            patch_size = (int(args.y_patch), int(args.x_patch))
        else:
            patch_size = (z, int(args.y_patch), int(args.x_patch))
    else:
        raise ValueError(f"Unsupported axes string: {axes}")
    return axes, patch_size


def try_set(o, path, value) -> bool:
    parts = path.split(".")
    cur = o
    for i, p in enumerate(parts):
        is_last = i == len(parts) - 1
        if hasattr(cur, p):
            if is_last:
                try:
                    setattr(cur, p, value)
                    return True
                except Exception:
                    return False
            cur = getattr(cur, p)
        elif isinstance(cur, dict) and p in cur:
            if is_last:
                cur[p] = value
                return True
            cur = cur[p]
        else:
            return False
    return False


def main():
    ap = argparse.ArgumentParser(description="Train CAREamics Noise2Void on TIFF patches.")
    ap.add_argument("--patch_dir", required=True)
    ap.add_argument("--save_dir",  required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--y_patch", type=int, default=64)
    ap.add_argument("--x_patch", type=int, default=64)
    ap.add_argument("--z_patch", type=int, default=1)
    ap.add_argument("--axes", default="auto", choices=["auto", "YX", "SYX", "ZYX", "SZYX"])
    ap.add_argument("--augment", dest="use_augmentations", action="store_true")
    ap.add_argument("--no_augment", dest="use_augmentations", action="store_false")
    ap.set_defaults(use_augmentations=False)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--skip_sanity", action="store_true")
    ap.add_argument("--export_bmz", action="store_true")
    ap.add_argument("--bmz_name", default="n2v_liver_model")
    ap.add_argument("--bmz_desc", default="Noise2Void model for GFP liver slice denoising.")
    args = ap.parse_args()

    is_windows = platform.system().lower().startswith("win")
    if args.num_workers is None:
        args.num_workers = 0 if is_windows else min(8, max(2, (os.cpu_count() or 8) // 2))

    patch_dir = Path(args.patch_dir)
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_patches_stack(patch_dir)  # (N, Y, X)
    print(f"[INFO] Loaded stack shape: {data.shape}")

    axes, patch_size = decide_axes_and_patchsize(data.ndim, args)
    print(f"[INFO] Training axes: {axes}, patch_size: {patch_size}")

    config = create_n2v_configuration(
        experiment_name=args.bmz_name,
        data_type="array",
        axes=axes,
        patch_size=patch_size,
        batch_size=int(args.batch_size),
        num_epochs=int(args.epochs),
        #use_augmentations=bool(args.use_augmentations),
    )
    print("[INFO] Created configuration.")

    set_flags = []
    for path, val in [
        ("data.num_workers", args.num_workers),
        ("data.loader_workers", args.num_workers),
        ("data.pin_memory", not is_windows),
        ("training.trainer.num_sanity_val_steps", 0 if args.skip_sanity else 2),
        ("trainer.num_sanity_val_steps", 0 if args.skip_sanity else 2),
        ("training.num_sanity_val_steps", 0 if args.skip_sanity else 2),
        ("data.persistent_workers", not is_windows),
    ]:
        if try_set(config, path, val):
            set_flags.append(f"{path}={val}")
    if set_flags:
        print("[INFO] Applied config tweaks: " + ", ".join(set_flags))

    # Train into save_dir
    careamist = CAREamist(source=config, work_dir=str(save_dir))
    careamist.train(train_source=data, val_percentage=0.0, val_minimum_split=10)

    # Best checkpoint path
    ckpt_cb = getattr(careamist.trainer, "checkpoint_callback", None)
    best_ckpt = getattr(ckpt_cb, "best_model_path", None)
    print(f"[INFO] Best checkpoint: {best_ckpt}")

    # Optional BMZ export — FIX: input_array shape must match axes
    if args.export_bmz:
        if axes.startswith("S"):     # 'SYX' or 'SZYX' -> needs sample dim
            export_input = data[:1]  # shape (1, Y, X) or (1, Z, Y, X)
        else:                         # 'YX' or 'ZYX' -> single sample without S
            export_input = data[0]    # shape (Y, X) or (Z, Y, X)

        export_zip = save_dir / f"{args.bmz_name}.zip"
        careamist.export_to_bmz(
            path_to_archive=export_zip,
            friendly_model_name=args.bmz_name,
            input_array=export_input,
            authors=[{"name": "Arkajyoti"}],
            general_description=args.bmz_desc,
            data_description=f"2D patches ({args.y_patch}x{args.x_patch}), single-channel.",
        )
        print(f"[DONE] Exported BMZ model: {export_zip.resolve()}")
    else:
        print("[DONE] Training complete (BMZ export skipped). See logs/checkpoints in save_dir.")


if __name__ == "__main__":
    main()
