# STS-Stega

Syndrome-Trellis Sampler for generative image steganography.

This project combines DDPM-based image generation with syndrome-trellis sampling (STS). The main idea is an asymmetric steganographic workflow: embedding uses a diffusion model and GPU-side probability estimates, while extraction only needs the parity-check matrix and XOR operations.

## Features

- Diffusion posterior probability extraction for pixel-level sampling.
- Syndrome-Trellis Sampler with Numba-accelerated forward/backward trellis passes.
- STC baseline sampler with multiple parity-check matrix construction methods.
- Batch embedding/extraction scripts for generating cover/stego pairs.
- Evaluation helpers for image quality and payload experiments.

## Repository layout

```text
.
├── guided_diffusion/          # DDPM implementation and model utilities
├── evaluate/                  # Evaluation scripts
├── H_matrix/                  # Precomputed parity-check matrices
├── ref_imgs/                  # Small reference image sets
├── scripts/                   # Diffusion training/sampling scripts
├── stscp_sampler_new.py       # Core STS sampler
├── stc_sampler.py             # STC sampler baseline
├── stega_stscp_new.py         # STS embedding/extraction entry point
├── stega_hybrid_new.py        # Legacy hybrid pipeline
├── run_batch_hybrid_new.py    # Batch generation pipeline
├── run_batch_stc.py           # STC batch pipeline
└── utils.py                   # Probability conversion and image helpers
```

## What is intentionally not tracked

The repository is configured to exclude large or generated local artifacts, including:

- model checkpoints under `models/` (`*.pt`, `*.pth`, `*.ckpt`, etc.),
- generated `output*` / `results*` experiment directories,
- generated payload and message files,
- local notes and assistant state files.

Place downloaded model checkpoints in `models/` locally before running generation scripts.

## Environment setup

Create a Python environment and install the expected scientific Python stack:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scipy pillow tqdm numba mpi4py blobfile
```

Depending on your CUDA version and GPU environment, install PyTorch from the official PyTorch index appropriate for your system.

## Basic usage

### Single-method scripts

The core embedding pipeline is implemented in:

```bash
python stega_stscp_new.py --help
```

The legacy hybrid variant is available via:

```bash
python stega_hybrid_new.py --help
```

### Batch experiments

Run the batch hybrid pipeline:

```bash
python run_batch_hybrid_new.py \
  --base-samples ref_imgs/face \
  --model-path models/ffhq_10m.pt \
  --num-images 10
```

Run the STC baseline batch pipeline:

```bash
python run_batch_stc.py --help
```

Outputs are written to local `output*` directories, which are ignored by git.

## Paper

The ACM CCS-style manuscript source is under `paper_latex/`.

A full LaTeX build normally requires:

```bash
cd paper_latex
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Generated PDFs and auxiliary files are intentionally ignored.

## Notes

- Extraction is designed to be black-box with respect to the diffusion model: it uses the parity-check matrix and stego bit planes.
- The full experimental workflow requires local model checkpoints that are not included in this repository.
- Large outputs and temporary evaluation caches should stay outside version control.
