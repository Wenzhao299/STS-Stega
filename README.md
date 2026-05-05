# STS-Stega

Official implementation of **STS-Stega: Asymmetric Generative Image Steganography via Syndrome-Trellis Sampling in Diffusion Models**.

STS-Stega is a generative image steganography framework that combines denoising diffusion models with syndrome-trellis sampling. The sender uses the diffusion model to estimate pixel-level posterior probabilities and performs constrained sampling, while the receiver extracts the payload with only the stego image and a shared parity-check matrix.

## Highlights

- **Asymmetric extraction.** Decoding does not require the diffusion model, GPU inference, or probability recomputation.
- **Syndrome-trellis sampling for diffusion models.** Pixel posteriors are converted into context-conditional bit-plane probabilities for payload-constrained sampling.
- **Bit-plane payload embedding.** RGB images are decomposed into 24 binary planes and embedded with entropy-aware payload allocation.
- **Reproducible baselines.** The repository includes STS and STC samplers, batch generation scripts, and evaluation utilities.

## Repository Structure

```text
.
├── guided_diffusion/          # Diffusion model implementation and utilities
├── scripts/                   # Diffusion training and sampling entry points
├── evaluate/                  # FID, PSNR, and SSIM evaluation scripts
├── H_matrix/                  # Precomputed parity-check matrices
├── ref_imgs/                  # Small reference image sets
├── stscp_sampler_new.py       # Core STS sampler
├── stc_sampler.py             # STC baseline sampler
├── stega_stscp_new.py         # Single-image STS embedding and extraction
├── stega_hybrid_new.py        # Hybrid experimental pipeline
├── run_batch_hybrid_new.py    # Batch STS generation and extraction
├── run_batch_stc.py           # Batch STC baseline
└── utils.py                   # Probability conversion and image helpers
```

## Installation

Create a Python environment and install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scipy pillow tqdm numba mpi4py blobfile
```

Install the PyTorch build that matches your CUDA driver and GPU environment. Model checkpoints are not included in this repository; place downloaded checkpoints under `models/` before running generation scripts.

## Usage

Inspect the single-image STS entry point:

```bash
python stega_stscp_new.py --help
```

Run a small batch STS experiment:

```bash
python run_batch_hybrid_new.py \
  --base-samples ref_imgs/face \
  --model-path models/ffhq_10m.pt \
  --num-images 10 \
  --output-root output_hybrid_parallel
```

Run the STC baseline:

```bash
python run_batch_stc.py --help
```

Generated cover, stego, message, and metadata files are written to local `output*` directories.

## Evaluation

Image quality metrics:

```bash
python evaluate/psnr_ssim_test.py --help
```

FID evaluation:

```bash
python evaluate/fid_test.py --help
```

The `evaluate/sts/` and `evaluate/stc/` directories include example JSONL outputs from STS and STC experiments.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{sts_stega,
  title = {STS-Stega: Asymmetric Generative Image Steganography via Syndrome-Trellis Sampling in Diffusion Models},
  booktitle = {Proceedings of the ACM Conference on Computer and Communications Security},
  year = {2026}
}
```

## Acknowledgements

This codebase builds on the DDPM implementation from guided diffusion and uses syndrome-trellis coding ideas for practical generative steganography.
