#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import warnings

import numpy as np
import tifffile as tiff
import torch
import yaml
from cellpose import models

warnings.filterwarnings("ignore", category=UserWarning, module="bfio")
warnings.filterwarnings("ignore", category=ImportWarning, module="aicsimageio")

# ---------------- utils ----------------

FNAME_RE = re.compile(
    r"""(?ix)
    ^(?P<prefix>.*?)
    (?P<well>[A-G]\d{2})
    f(?P<site>\d{2,3})
    d(?P<chan>\d{1,2})
    \.(?:tif|tiff)$
    """
)

def parse_name(p: Path):
    m = FNAME_RE.match(p.name)
    if not m:
        return None
    gd = m.groupdict()
    return gd["prefix"], gd["well"], int(gd["site"]), int(gd["chan"])

def read_tiff_robust(p: Path) -> np.ndarray:
    """Read TIFF while ignoring inconsistent OME XML. Returns (Z?, Y, X) or (Y, X)."""
    try:
        return np.asarray(tiff.imread(str(p)))
    except Exception:
        with tiff.TiffFile(str(p)) as tf:
            if len(tf.pages) == 1:
                return tf.pages[0].asarray()
            pages = [pg.asarray() for pg in tf.pages]
        return np.stack(pages, axis=0)

def z_project(stack: np.ndarray, strategy: str = "max") -> np.ndarray:
    """Project any >=3D stack down to 2D (YX)."""
    if stack.ndim == 2:
        return stack
    proj = stack
    while proj.ndim > 2:
        proj = np.median(proj, axis=0) if strategy == "median" else np.max(proj, axis=0)
    return proj

def normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    p1, p99 = np.percentile(img, (1, 99))
    if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
        img = (img - p1) / (p99 - p1)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0)

def to_u16(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)

def run_cellpose(img2d: np.ndarray, model_identifier: str | None, use_gpu: bool,
                 flow_threshold: float | None, cellprob_threshold: float | None) -> np.ndarray:
    gpu_ok = bool(use_gpu and torch.cuda.is_available())
    mdl = models.CellposeModel(model_type=model_identifier, gpu=gpu_ok) if model_identifier else models.CellposeModel(gpu=gpu_ok)
    kwargs = {}
    if flow_threshold is not None:
        kwargs["flow_threshold"] = float(flow_threshold)
    if cellprob_threshold is not None:
        kwargs["cellprob_threshold"] = float(cellprob_threshold)
    masks, *_ = mdl.eval(img2d, **kwargs)
    return masks.astype(np.uint16)

# ---------------- main ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Segment channels selected by filename tokens (…A03f00dX.TIF).")
    p.add_argument("config", help="Path to YAML config.")
    # Optional CLI overrides (aligned with d0 / d3 naming)
    p.add_argument("--d0-model", type=str, default=None)
    p.add_argument("--d3-model", type=str, default=None)
    p.add_argument("--d0-flow", type=float, default=None)
    p.add_argument("--d0-cellprob", type=float, default=None)
    p.add_argument("--d3-flow", type=float, default=None)
    p.add_argument("--d3-cellprob", type=float, default=None)
    p.add_argument("--zstack", dest="zstk", action="store_true")
    p.add_argument("--no-zstack", dest="zstk", action="store_false")
    p.set_defaults(zstk=None)
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    in_dir = Path(cfg["input_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    z_strategy = cfg.get("z_strategy", "max")
    excludes = [e.lower() for e in cfg.get("exclude_name_contains", [])]

    cp_cfg = cfg.get("cellpose", {})
    use_gpu = bool(cp_cfg.get("use_gpu", True))

    ch_cfg = cfg.get("channels", {})
    d0_cfg  = ch_cfg.get("d0", {})
    d3_cfg  = ch_cfg.get("d3", {})

    # filename channel ids to use (e.g., d0 -> 0, d3 -> 3)
    d0_ch  = int(d0_cfg.get("filename_channel_id"))
    d3_ch  = int(d3_cfg.get("filename_channel_id"))

    # models & thresholds (CLI > YAML)
    d0_model = args.d0_model if args.d0_model is not None else d0_cfg.get("model", "nuclei")
    d3_model = args.d3_model if args.d3_model is not None else d3_cfg.get("model")
    d0_flow  = args.d0_flow  if args.d0_flow  is not None else d0_cfg.get("flow_threshold")
    d0_prob  = args.d0_cellprob if args.d0_cellprob is not None else d0_cfg.get("cellprob_threshold")
    d3_flow  = args.d3_flow  if args.d3_flow  is not None else d3_cfg.get("flow_threshold")
    d3_prob  = args.d3_cellprob if args.d3_cellprob is not None else d3_cfg.get("cellprob_threshold")

    save_cfg = cfg.get("save", {})
    save_zstk = args.zstk if args.zstk is not None else bool(save_cfg.get("zstack", False))

    # collect all tif/tiff files
    all_files = [p for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF") for p in in_dir.rglob(ext)]
    files = [p for p in all_files if not any(x in p.name.lower() for x in excludes)]
    if not files:
        print(f"[INFO] No TIFF files found under {in_dir}")
        return

    # group by (relative directory, prefix, well, site)
    groups = {}
    for p in files:
        parsed = parse_name(p)
        if not parsed:
            print(f"[WARN] Skip (name pattern mismatch): {p.name}")
            continue
        prefix, well, site, chan = parsed
        key = (p.relative_to(in_dir).parent, prefix, well, site)
        groups.setdefault(key, {})[chan] = p

    print(f"[INFO] Found {len(groups)} (well,site) groups")
    print(f"[INFO] GPU available: {torch.cuda.is_available()}  |  Use GPU: {use_gpu}")
    print(f"[INFO] save.zstack={save_zstk}")
    print(f"[INFO] d0 -> model={d0_model} (flow,cellprob)=({d0_flow},{d0_prob})")
    print(f"[INFO] d3 -> model={d3_model} (flow,cellprob)=({d3_flow},{d3_prob})")

    for (rel_dir, prefix, well, site), chans in groups.items():
        if d0_ch not in chans or d3_ch not in chans:
            missing = []
            if d0_ch not in chans: missing.append(f"d{d0_ch}")
            if d3_ch not in chans: missing.append(f"d{d3_ch}")
            print(f"[WARN] Missing {','.join(missing)} for {well} f{site:02d} -> skipping")
            continue

        # paths for two channels (keep stems with original dX)
        d0_path = chans[d0_ch]
        d3_path = chans[d3_ch]
        d0_stem = d0_path.stem     # ...A03f00d0
        d3_stem = d3_path.stem     # ...A03f00d3

        # robust reads (avoid OME series mismatch)
        d0_arr = read_tiff_robust(d0_path)
        d3_arr = read_tiff_robust(d3_path)

        # projection + normalize
        d0_proj = normalize01(z_project(d0_arr, z_strategy))
        d3_proj = normalize01(z_project(d3_arr, z_strategy))

        # segment
        d0_masks = run_cellpose(d0_proj, d0_model, use_gpu, d0_flow, d0_prob)
        d3_masks = run_cellpose(d3_proj, d3_model, use_gpu, d3_flow, d3_prob)

        # output directory mirrors input
        sub_out = (out_dir / rel_dir)
        sub_out.mkdir(parents=True, exist_ok=True)

        # preserve original dX token in filenames
        tiff.imwrite(sub_out / f"{d0_stem}_cp.tif", d0_masks)
        tiff.imwrite(sub_out / f"{d3_stem}_cp.tif", d3_masks)

        if save_zstk:
            if d0_arr.ndim > 2:
                tiff.imwrite(sub_out / f"{d0_stem}_z.tif", to_u16(normalize01(d0_arr)))
            if d3_arr.ndim > 2:
                tiff.imwrite(sub_out / f"{d3_stem}_z.tif", to_u16(normalize01(d3_arr)))

        print(f"[OK] {rel_dir}/{well} f{site:02d}")

    print(f"\n[FINISHED] Results written under: {out_dir}\n")

if __name__ == "__main__":
    main()
