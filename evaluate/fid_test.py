#!/usr/bin/env python3
"""FID evaluation for cover/stega images against a real-image dataset.

The script:
1. matches `cover_x.png` with `stega_x.png`,
2. writes one JSON object per pair using the pairwise feature distance,
3. recursively traverses a real FFHQ image root such as
   `/data/home/wls_cwz/data/dataset/ffhq/images256x256/<folder>/xxx.png`,
4. reports standard FID scores for `cover vs real`, `stega vs real`, and
   `cover vs stega`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


FID_CKPT_NAME = "pt_inception-2015-12-05-6726825d.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cover/stega FID scores and save JSONL."
    )
    parser.add_argument(
        "--cover-dir",
        type=Path,
        default=Path("output_marginal_ablation/ffhq-7-6bpp/cover"),
        help="Directory containing cover_x.png files.",
    )
    parser.add_argument(
        "--stega-dir",
        type=Path,
        default=Path("output_marginal_ablation/ffhq-7-6bpp/stega"),
        help="Directory containing stega_x.png files.",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=Path("/data/home/wls_cwz/data/dataset/ffhq/images256x256/00000/"),
        help="Root directory of real images. Subfolders will be traversed recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to evaluate/<dataset_name>_fid.jsonl.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for Inception feature extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:3",
        help="Device to use, e.g. cpu, cuda, cuda:0. Defaults to auto.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        choices=(64, 192, 768, 2048),
        help="Inception feature dimensionality.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N matched cover/stega pairs. Useful for debugging.",
    )
    parser.add_argument(
        "--real-limit",
        type=int,
        default=None,
        help="Only process the first N real images after recursive traversal. Useful for debugging.",
    )
    parser.add_argument(
        "--real-image-size",
        type=int,
        default=256,
        help="Resize each real image to this square size before feature extraction. Defaults to 256.",
    )
    parser.add_argument(
        "--fid-ckpt",
        type=Path,
        default=None,
        help="Local path to pt_inception FID checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--real-stats-cache",
        type=Path,
        default=None,
        help="Path to cached real-image FID statistics (.npz). Defaults to evaluate/fid_cache/<auto>.npz.",
    )
    parser.add_argument(
        "--refresh-real-stats",
        action="store_true",
        help="Recompute and overwrite cached real-image statistics even if a matching cache exists.",
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        help="Precompute real-image statistics and save them as a pytorch-fid-compatible .npz, then exit.",
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
    # type_name = cover_dir.name.split("_")[1]
    script_dir = Path(__file__).resolve().parent
    return script_dir / f"sts/ablation_{dataset_name}_fid.jsonl"


def collect_real_images(real_dir: Path, limit: int | None = None) -> list[Path]:
    if not real_dir.exists():
        raise FileNotFoundError(f"Real-image directory not found: {real_dir}")

    image_paths = sorted(path for path in real_dir.rglob("*.png") if path.is_file())
    if not image_paths:
        raise RuntimeError(f"No PNG images found under real-image root: {real_dir}")

    if limit is not None:
        image_paths = image_paths[:limit]

    return image_paths


def load_statistics_from_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Statistics archive not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        if "mu" not in data or "sigma" not in data:
            raise KeyError(f"Statistics archive must contain 'mu' and 'sigma': {npz_path}")
        mu = data["mu"]
        sigma = data["sigma"]
        count = int(data["count"].item()) if "count" in data.files else -1
    return mu, sigma, count


def slugify_path_name(path: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name or "dataset").strip("._")
    return slug or "dataset"


def resolve_real_stats_cache_path(
    real_dir: Path,
    dims: int,
    real_limit: int | None,
    real_image_size: int,
    requested_cache: Path | None,
) -> Path:
    if requested_cache is not None:
        return requested_cache

    script_dir = Path(__file__).resolve().parent
    cache_dir = script_dir / "fid_cache"
    dir_hash = hashlib.sha256(str(real_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    limit_tag = "all" if real_limit is None else str(real_limit)
    filename = (
        f"{slugify_path_name(real_dir)}_real{real_image_size}_dims{dims}_limit{limit_tag}_{dir_hash}.npz"
    )
    return cache_dir / filename


def fingerprint_image_paths(image_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for image_path in image_paths:
        stat = image_path.stat()
        digest.update(str(image_path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_cached_real_statistics(
    cache_path: Path,
    real_dir: Path,
    dims: int,
    real_limit: int | None,
    real_image_size: int,
    fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as data:
        cached_real_dir = str(data["real_dir"].item())
        cached_dims = int(data["dims"].item())
        cached_real_limit = int(data["real_limit"].item())
        cached_real_image_size = int(data["real_image_size"].item()) if "real_image_size" in data.files else -1
        cached_fingerprint = str(data["fingerprint"].item())

        requested_limit = -1 if real_limit is None else int(real_limit)
        if cached_real_dir != str(real_dir.resolve()):
            return None
        if cached_dims != dims:
            return None
        if cached_real_limit != requested_limit:
            return None
        if cached_real_image_size != int(real_image_size):
            return None
        if cached_fingerprint != fingerprint:
            return None

        mu = data["mu"]
        sigma = data["sigma"]
        count = int(data["count"].item())
        return mu, sigma, count


def save_real_statistics_cache(
    cache_path: Path,
    real_dir: Path,
    dims: int,
    real_limit: int | None,
    real_image_size: int,
    fingerprint: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    count: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        mu=mu,
        sigma=sigma,
        count=np.int64(count),
        dims=np.int64(dims),
        real_limit=np.int64(-1 if real_limit is None else real_limit),
        real_image_size=np.int64(real_image_size),
        real_dir=np.array(str(real_dir.resolve())),
        fingerprint=np.array(fingerprint),
    )


def get_or_compute_real_statistics(
    image_paths: list[Path],
    model: InceptionV3,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    dims: int,
    real_dir: Path,
    real_limit: int | None,
    real_image_size: int,
    cache_path: Path,
    refresh_cache: bool,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    fingerprint = fingerprint_image_paths(image_paths)

    if not refresh_cache:
        cached_stats = load_cached_real_statistics(
            cache_path=cache_path,
            real_dir=real_dir,
            dims=dims,
            real_limit=real_limit,
            real_image_size=real_image_size,
            fingerprint=fingerprint,
        )
        if cached_stats is not None:
            mu, sigma, count = cached_stats
            print(f"Loaded cached real features from: {cache_path}")
            return mu, sigma, count, True

    mu, sigma, count = extract_statistics_for_paths(
        image_paths=image_paths,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        dims=dims,
        desc="Extracting real features",
        image_size=real_image_size,
    )
    save_real_statistics_cache(
        cache_path=cache_path,
        real_dir=real_dir,
        dims=dims,
        real_limit=real_limit,
        real_image_size=real_image_size,
        fingerprint=fingerprint,
        mu=mu,
        sigma=sigma,
        count=count,
    )
    print(f"Saved cached real features to: {cache_path}")
    return mu, sigma, count, False


def resolve_fid_checkpoint(requested_path: Path | None) -> Path:
    candidates: list[Path] = []
    if requested_path is not None:
        candidates.append(requested_path)

    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        candidates.append(Path(torch_home) / "hub" / "checkpoints" / FID_CKPT_NAME)

    candidates.extend(
        [
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / FID_CKPT_NAME,
            Path("/data/home/wls_cwz/.cache/torch/hub/checkpoints") / FID_CKPT_NAME,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to find the local FID Inception checkpoint. "
        "Please pass --fid-ckpt explicitly."
    )


class PairedImageDataset(Dataset):
    def __init__(self, pairs: list[tuple[int, Path, Path]]) -> None:
        self.pairs = pairs
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_id, cover_path, stega_path = self.pairs[index]
        cover_img = Image.open(cover_path).convert("RGB")
        stega_img = Image.open(stega_path).convert("RGB")
        return (
            self.to_tensor(cover_img),
            self.to_tensor(stega_img),
            image_id,
            str(cover_path),
            str(stega_path),
        )


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_size: int | None = None) -> None:
        self.image_paths = image_paths
        self.image_size = image_size
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return self.to_tensor(image), str(image_path)


class FeatureStatsAccumulator:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.count = 0
        self.sum = np.zeros(dims, dtype=np.float64)
        self.sum_outer = np.zeros((dims, dims), dtype=np.float64)

    def update(self, activations: np.ndarray) -> None:
        if activations.ndim != 2 or activations.shape[1] != self.dims:
            raise ValueError(
                f"Expected activations with shape (N, {self.dims}), got {activations.shape}"
            )

        batch = activations.astype(np.float64, copy=False)
        self.count += batch.shape[0]
        self.sum += np.sum(batch, axis=0)
        self.sum_outer += batch.T @ batch

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("Cannot finalize feature statistics with zero samples.")

        mu = self.sum / self.count
        if self.count < 2:
            sigma = np.zeros((self.dims, self.dims), dtype=np.float64)
        else:
            sigma = (self.sum_outer - self.count * np.outer(mu, mu)) / (self.count - 1)
        return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Mean vectors have different lengths.")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Covariance matrices have different shapes.")

    diff = mu1 - mu2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=linalg.LinAlgWarning)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=linalg.LinAlgWarning)
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component in covmean: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_frechet_distance_with_logging(
    label: str,
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    start = time.perf_counter()
    print(f"Computing {label} FID...", flush=True)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    elapsed = time.perf_counter() - start
    print(f"Finished {label} FID in {elapsed:.2f}s", flush=True)
    return fid_value


def _inception_v3(*args, **kwargs):
    try:
        version = tuple(map(int, torchvision.__version__.split(".")[:2]))
    except ValueError:
        version = (0,)

    if version >= (0, 6):
        kwargs["init_weights"] = False

    if version < (0, 13) and "weights" in kwargs:
        if kwargs["weights"] == "DEFAULT":
            kwargs["pretrained"] = True
        elif kwargs["weights"] is None:
            kwargs["pretrained"] = False
        else:
            raise ValueError(
                f"weights=={kwargs['weights']} not supported in torchvision {torchvision.__version__}"
            )
        del kwargs["weights"]

    return torchvision.models.inception_v3(*args, **kwargs)


class InceptionV3(nn.Module):
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {64: 0, 192: 1, 768: 2, 2048: 3}

    def __init__(
        self,
        output_blocks: Iterable[int] = (DEFAULT_BLOCK_INDEX,),
        resize_input: bool = True,
        normalize_input: bool = True,
        requires_grad: bool = False,
        fid_ckpt: Path | None = None,
    ) -> None:
        super().__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(self.output_blocks)
        if self.last_needed_block > 3:
            raise ValueError("Last possible output block index is 3.")

        inception = fid_inception_v3(fid_ckpt)
        self.blocks = nn.ModuleList()

        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outputs.append(x)
            if idx == self.last_needed_block:
                break

        return outputs


def fid_inception_v3(fid_ckpt: Path | None) -> nn.Module:
    inception = _inception_v3(num_classes=1008, aux_logits=False, weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    if fid_ckpt is None:
        raise ValueError("fid_ckpt must not be None.")
    state_dict = torch.load(fid_ckpt, map_location="cpu", weights_only=True)
    inception.load_state_dict(state_dict)
    inception.eval()
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__(in_channels, pool_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    def __init__(self, in_channels: int, channels_7x7: int) -> None:
        super().__init__(in_channels, channels_7x7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


def build_progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit="batch")


def get_inception_features(model: InceptionV3, images: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        pred = model(images)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    return pred.squeeze(3).squeeze(2).cpu().numpy()


def extract_statistics_for_paths(
    image_paths: list[Path],
    model: InceptionV3,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    dims: int,
    desc: str,
    image_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    dataset = ImagePathDataset(image_paths, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    stats = FeatureStatsAccumulator(dims)

    for image_batch, _ in build_progress(dataloader, desc):
        image_batch = image_batch.to(device, non_blocking=True)
        activations = get_inception_features(model, image_batch)
        stats.update(activations)

    mu, sigma = stats.finalize()
    return mu, sigma, stats.count


def evaluate_pairs(
    pairs: list[tuple[int, Path, Path]],
    real_images: list[Path],
    real_root: Path,
    real_limit: int | None,
    real_image_size: int,
    model: InceptionV3,
    device: torch.device,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    dims: int,
    real_stats_cache_path: Path,
    refresh_real_stats: bool,
    real_stats_override: tuple[np.ndarray, np.ndarray, int] | None = None,
    real_stats_source: str | None = None,
) -> dict[str, float | int | str]:
    dataset = PairedImageDataset(pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    cover_stats = FeatureStatsAccumulator(dims)
    stega_stats = FeatureStatsAccumulator(dims)
    pair_fid_sum = 0.0
    pair_count = 0
    dataset_name = pairs[0][1].parent.parent.name

    with output_path.open("w", encoding="utf-8") as writer:
        for cover_batch, stega_batch, image_ids, cover_paths, stega_paths in build_progress(
            dataloader, "Extracting features"
        ):
            cover_batch = cover_batch.to(device, non_blocking=True)
            stega_batch = stega_batch.to(device, non_blocking=True)

            cover_np = get_inception_features(model, cover_batch)
            stega_np = get_inception_features(model, stega_batch)
            cover_stats.update(cover_np)
            stega_stats.update(stega_np)

            pair_fids = np.sum((cover_np - stega_np) ** 2, axis=1)
            image_ids_list = image_ids.tolist() if hasattr(image_ids, "tolist") else list(image_ids)

            for image_id, cover_path, stega_path, pair_fid in zip(
                image_ids_list, cover_paths, stega_paths, pair_fids
            ):
                writer.write(
                    json.dumps(
                        {
                            "type": "pair",
                            "dataset": dataset_name,
                            "image_id": int(image_id),
                            "cover": cover_path,
                            "stega": stega_path,
                            "fid": float(pair_fid),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                pair_fid_sum += float(pair_fid)
                pair_count += 1

        mu_cover, sigma_cover = cover_stats.finalize()
        mu_stega, sigma_stega = stega_stats.finalize()
        if real_stats_override is not None:
            mu_real, sigma_real, real_count = real_stats_override
            real_stats_cache_used = True
            if real_stats_source is not None:
                print(f"Loaded real statistics from archive: {real_stats_source}", flush=True)
        else:
            mu_real, sigma_real, real_count, real_stats_cache_used = get_or_compute_real_statistics(
                image_paths=real_images,
                model=model,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                dims=dims,
                real_dir=real_root,
                real_limit=real_limit,
                real_image_size=real_image_size,
                cache_path=real_stats_cache_path,
                refresh_cache=refresh_real_stats,
            )
        global_fid = calculate_frechet_distance_with_logging(
            "cover_vs_stega",
            mu_cover,
            sigma_cover,
            mu_stega,
            sigma_stega,
        )
        mean_fid = pair_fid_sum / max(pair_count, 1)
        cover_vs_real_fid = calculate_frechet_distance_with_logging(
            "cover_vs_real",
            mu_cover,
            sigma_cover,
            mu_real,
            sigma_real,
        )
        stega_vs_real_fid = calculate_frechet_distance_with_logging(
            "stega_vs_real",
            mu_stega,
            sigma_stega,
            mu_real,
            sigma_real,
        )

        summary = {
            "type": "summary",
            "dataset": dataset_name,
            "count": pair_count,
            "real_count": real_count,
            "dims": dims,
            "real_image_size": real_image_size,
            "mean_fid": float(mean_fid),
            "global_fid": float(global_fid),
            "cover_vs_real_fid": float(cover_vs_real_fid),
            "stega_vs_real_fid": float(stega_vs_real_fid),
            "real_dir": str(real_root),
            "real_stats_cache": str(real_stats_cache_path),
            "real_stats_source": real_stats_source or str(real_stats_cache_path),
            "real_stats_cache_used": bool(real_stats_cache_used),
            "output": str(output_path),
        }
        writer.write(json.dumps(summary, ensure_ascii=False) + "\n")

    return summary


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    fid_ckpt = resolve_fid_checkpoint(args.fid_ckpt)
    real_stats_cache_path = resolve_real_stats_cache_path(
        real_dir=args.real_dir,
        dims=args.dims,
        real_limit=args.real_limit,
        real_image_size=args.real_image_size,
        requested_cache=args.real_stats_cache,
    )

    if args.save_stats:
        if args.real_dir.suffix.lower() == ".npz":
            raise ValueError("--save-stats expects --real-dir to be an image directory, not an .npz file.")

        real_images = collect_real_images(args.real_dir, limit=args.real_limit)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
        model = InceptionV3(output_blocks=(block_idx,), fid_ckpt=fid_ckpt).to(device)
        model.eval()

        mu_real, sigma_real, real_count, cache_used = get_or_compute_real_statistics(
            image_paths=real_images,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dims=args.dims,
            real_dir=args.real_dir,
            real_limit=args.real_limit,
            real_image_size=args.real_image_size,
            cache_path=real_stats_cache_path,
            refresh_cache=args.refresh_real_stats,
        )
        summary = {
            "type": "real_stats",
            "dims": args.dims,
            "real_count": real_count,
            "real_image_size": args.real_image_size,
            "real_dir": str(args.real_dir),
            "real_stats_cache": str(real_stats_cache_path),
            "real_stats_cache_used": bool(cache_used),
            "mu_shape": list(mu_real.shape),
            "sigma_shape": list(sigma_real.shape),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    pairs = build_pairs(args.cover_dir, args.stega_dir, limit=args.limit)
    output_path = resolve_output_path(args.cover_dir, args.output)

    real_images: list[Path] = []
    real_stats_override: tuple[np.ndarray, np.ndarray, int] | None = None
    real_stats_source: str | None = None
    if args.real_dir.suffix.lower() == ".npz":
        mu_real, sigma_real, real_count = load_statistics_from_npz(args.real_dir)
        real_stats_override = (mu_real, sigma_real, real_count)
        real_stats_source = str(args.real_dir)
    else:
        real_images = collect_real_images(args.real_dir, limit=args.real_limit)
        real_stats_source = str(real_stats_cache_path)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3(output_blocks=(block_idx,), fid_ckpt=fid_ckpt).to(device)
    model.eval()

    summary = evaluate_pairs(
        pairs=pairs,
        real_images=real_images,
        real_root=args.real_dir,
        real_limit=args.real_limit,
        real_image_size=args.real_image_size,
        model=model,
        device=device,
        output_path=output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dims=args.dims,
        real_stats_cache_path=real_stats_cache_path,
        refresh_real_stats=args.refresh_real_stats,
        real_stats_override=real_stats_override,
        real_stats_source=real_stats_source,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
