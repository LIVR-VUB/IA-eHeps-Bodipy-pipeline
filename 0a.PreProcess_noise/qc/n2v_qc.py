#!/usr/bin/env python3
"""
Quick QC comparing raw vs denoised image:
- Intensity histograms
- Side-by-side panels
- Difference heatmap
- SSIM metric

Usage:
  python qc_denoising.py \
      --raw /path/to/raw.tif \
      --denoised /path/to/raw_denoised.tif \
      --output ./qc_report
"""

import argparse
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def _to_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
        x = x[0] if x.shape[0] > 1 else x.max(axis=0)
    elif x.ndim == 3:
        x = x[0] if x.shape[0] > 1 else (x if x.shape[0] == 1 else x.max(axis=0))
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported shape: {x.shape}")
    return x


def run_qc(raw_path: Path, den_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _to_2d(imread(str(raw_path)))
    den = _to_2d(imread(str(den_path)))

    # Align shapes if needed (simple crop to min common)
    H = min(raw.shape[0], den.shape[0])
    W = min(raw.shape[1], den.shape[1])
    raw = raw[:H, :W]
    den = den[:H, :W]

    # Save aligned copies
    imwrite(str(out_dir / "raw_aligned.tif"), raw.astype(np.uint16), photometric="minisblack")
    imwrite(str(out_dir / "denoised_aligned.tif"), den.astype(np.uint16), photometric="minisblack")

    # Histograms
    plt.figure(figsize=(10, 4))
    plt.hist(raw.flatten(), bins=256, alpha=0.6, label="Raw", density=True)
    plt.hist(den.flatten(), bins=256, alpha=0.6, label="Denoised", density=True)
    plt.legend()
    plt.title("Pixel Intensity Distributions")
    plt.xlabel("Intensity")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_dir / "histogram.png", dpi=200)
    plt.close()

    # Panels
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(raw, cmap="gray")
    axs[0].set_title("Raw")
    axs[1].imshow(den, cmap="gray")
    axs[1].set_title("Denoised")

    # Difference heatmap (clip for visibility)
    diff = raw.astype(np.int32) - den.astype(np.int32)
    vmax = np.percentile(np.abs(diff), 99)
    axs[2].imshow(np.clip(diff, -vmax, vmax), cmap="seismic", vmin=-vmax, vmax=vmax)
    axs[2].set_title("Difference (Raw - Denoised)")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=200)
    plt.close()

    # SSIM (convert to float, normalize)
    r = raw.astype(np.float32)
    d = den.astype(np.float32)
    if r.max() > 0: r = r / r.max()
    if d.max() > 0: d = d / d.max()
    score = ssim(r, d, data_range=1.0)
    (out_dir / "metrics.txt").write_text(f"SSIM: {score:.4f}\n")
    print(f"[QC] SSIM = {score:.4f} (saved to metrics.txt)")


def main():
    ap = argparse.ArgumentParser(description="QC for denoising results.")
    ap.add_argument("--raw", required=True, help="Raw image path")
    ap.add_argument("--denoised", required=True, help="Denoised image path")
    ap.add_argument("--output", default="./qc_report", help="Output folder")
    args = ap.parse_args()

    run_qc(Path(args.raw), Path(args.denoised), Path(args.output))


if __name__ == "__main__":
    main()
