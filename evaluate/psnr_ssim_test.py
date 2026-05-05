#!/usr/bin/env python3
"""PSNR/SSIM evaluation for matched cover/stega image pairs.

The script:
1. matches `cover_x.png` with `stega_x.png`,
2. computes PSNR and SSIM for each pair,
3. writes one JSON object per pair,
4. writes a final summary JSON object with dataset-level averages.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR and SSIM for cover/stega image pairs and save JSONL."
    )
    parser.add_argument(
        "--cover-dir",
        type=Path,
        default=Path("output_hybrid_parallel/2026-03-19_16-43-15/cover"),
        help="Directory containing cover_x.png files.",
    )
    parser.add_argument(
        "--stega-dir",
        type=Path,
        default=Path("output_hybrid_parallel/2026-03-19_16-43-15/stega"),
        help="Directory containing stega_x.png files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to evaluate/<dataset_name>_psnr_ssim.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N matched cover/stega pairs. Useful for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def extract_image_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Failed to parse trailing numeric id from: {path}")
    return int(match.group(1))


def collect_indexed_files(directory: Path, prefix: str) -> dict[int, Path]:
    pattern = f"{prefix}_*.png"
    files: dict[int, Path] = {}
    for path in directory.glob(pattern):
        image_id = extract_image_id(path)
        files[image_id] = path
    return files


def build_pairs(
    cover_dir: Path,
    stega_dir: Path,
    limit: int | None = None,
) -> list[tuple[int, Path, Path]]:
    if not cover_dir.exists():
        raise FileNotFoundError(f"Cover directory not found: {cover_dir}")
    if not stega_dir.exists():
        raise FileNotFoundError(f"Stega directory not found: {stega_dir}")

    cover_files = collect_indexed_files(cover_dir, "cover")
    stega_files = collect_indexed_files(stega_dir, "stega")

    common_ids = sorted(set(cover_files) & set(stega_files))
    if not common_ids:
        raise RuntimeError("No matched cover/stega image pairs were found.")

    missing_cover = sorted(set(stega_files) - set(cover_files))
    missing_stega = sorted(set(cover_files) - set(stega_files))
    if missing_cover or missing_stega:
        print(
            "Warning: unmatched ids detected. "
            f"missing_cover={len(missing_cover)}, missing_stega={len(missing_stega)}"
        )

    if limit is not None:
        common_ids = common_ids[:limit]

    return [(image_id, cover_files[image_id], stega_files[image_id]) for image_id in common_ids]


def resolve_output_path(
    cover_dir: Path,
    requested_output: Path | None,
) -> Path:
    if requested_output is not None:
        return requested_output

    dataset_name = cover_dir.parent.name
    script_dir = Path(__file__).resolve().parent
    return script_dir / f"{dataset_name}_psnr_ssim.jsonl"


def build_progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit="pair")


def load_rgb_image(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64)


def calculate_psnr(image_a: np.ndarray, image_b: np.ndarray, data_range: float = 255.0) -> float:
    mse = np.mean((image_a - image_b) ** 2, dtype=np.float64)
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def calculate_ssim_single_channel(
    image_a: np.ndarray,
    image_b: np.ndarray,
    data_range: float = 255.0,
    sigma: float = 1.5,
) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_a = gaussian_filter(image_a, sigma=sigma, truncate=3.5)
    mu_b = gaussian_filter(image_b, sigma=sigma, truncate=3.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = gaussian_filter(image_a * image_a, sigma=sigma, truncate=3.5) - mu_a_sq
    sigma_b_sq = gaussian_filter(image_b * image_b, sigma=sigma, truncate=3.5) - mu_b_sq
    sigma_ab = gaussian_filter(image_a * image_b, sigma=sigma, truncate=3.5) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(ssim_map, dtype=np.float64))


def calculate_ssim(image_a: np.ndarray, image_b: np.ndarray, data_range: float = 255.0) -> float:
    if image_a.ndim != 3 or image_a.shape[2] != 3:
        raise ValueError(f"Expected RGB images with shape (H, W, 3), got {image_a.shape}")
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"PSNR/SSIM require identical image shapes, got {image_a.shape} vs {image_b.shape}"
        )

    channel_scores = [
        calculate_ssim_single_channel(image_a[:, :, channel], image_b[:, :, channel], data_range=data_range)
        for channel in range(3)
    ]
    return float(np.mean(channel_scores, dtype=np.float64))


def evaluate_pairs(
    pairs: list[tuple[int, Path, Path]],
    output_path: Path,
) -> dict[str, float | int | str]:
    dataset_name = pairs[0][1].parent.parent.name
    psnr_sum = 0.0
    ssim_sum = 0.0
    pair_count = 0

    with output_path.open("w", encoding="utf-8") as writer:
        for image_id, cover_path, stega_path in build_progress(pairs, "Evaluating PSNR/SSIM"):
            cover = load_rgb_image(cover_path)
            stega = load_rgb_image(stega_path)

            if cover.shape != stega.shape:
                raise ValueError(
                    f"Image shape mismatch for id={image_id}: "
                    f"{cover_path} -> {cover.shape}, {stega_path} -> {stega.shape}"
                )

            psnr_value = calculate_psnr(cover, stega)
            ssim_value = calculate_ssim(cover, stega)

            writer.write(
                json.dumps(
                    {
                        "type": "pair",
                        "dataset": dataset_name,
                        "image_id": int(image_id),
                        "cover": str(cover_path),
                        "stega": str(stega_path),
                        "psnr": float(psnr_value),
                        "ssim": float(ssim_value),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            psnr_sum += float(psnr_value)
            ssim_sum += float(ssim_value)
            pair_count += 1

        summary = {
            "type": "summary",
            "dataset": dataset_name,
            "count": pair_count,
            "mean_psnr": float(psnr_sum / max(pair_count, 1)),
            "mean_ssim": float(ssim_sum / max(pair_count, 1)),
            "output": str(output_path),
        }
        writer.write(json.dumps(summary, ensure_ascii=False) + "\n")

    return summary


def main() -> None:
    args = parse_args()

    pairs = build_pairs(args.cover_dir, args.stega_dir, limit=args.limit)
    output_path = resolve_output_path(args.cover_dir, args.output)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = evaluate_pairs(pairs=pairs, output_path=output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
