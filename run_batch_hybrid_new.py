"""
run_batch_hybrid.py

A batch processing script specifically for the 'hybrid' steganography method.
It uses the producer-consumer model to generate a dataset of cover/stega pairs,
where the steganography is performed by `stega_hybrid.py`.
"""

import argparse
import os
import multiprocessing
import multiprocessing.pool
import time
import re
import shutil
import json
import copy
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import glob # Added for file count checking
from PIL import Image

# --- Import from our project ---
from guided_diffusion import logger
from stscp_sampler_new import STSSampler

_HYBRID_BINDINGS: Optional[Dict[str, Any]] = None


def _load_hybrid_bindings() -> Dict[str, Any]:
    global _HYBRID_BINDINGS
    if _HYBRID_BINDINGS is None:
        from stega_hybrid_new import (
            create_argparser,
            embed_using_precalculated_data_hybrid,
            prepare_data_for_pair,
            resolve_model_preset,
        )

        _HYBRID_BINDINGS = {
            "create_argparser": create_argparser,
            "embed_using_precalculated_data_hybrid": embed_using_precalculated_data_hybrid,
            "prepare_data_for_pair": prepare_data_for_pair,
            "resolve_model_preset": resolve_model_preset,
        }
    return _HYBRID_BINDINGS

# --- Custom Non-Daemonic Pool for Nested Parallelism ---
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self): return False
    @daemon.setter
    def daemon(self, value): pass

class NoDaemonContext(type(multiprocessing.get_context("spawn"))):
    Process = NoDaemonProcess

class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


_BPP_DENOMINATOR = 65536


def _sanitize_name_fragment(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized.strip("._-") or "unknown_model"


def _infer_model_category(args) -> str:
    model_path = getattr(args, "model_path", "")
    if model_path:
        model_stem = os.path.splitext(os.path.basename(model_path))[0]
        model_stem = re.sub(r"_[0-9]+(?:[kKmMgG])?$", "", model_stem)
        if model_stem:
            return _sanitize_name_fragment(model_stem)

    preset_name = getattr(args, "model_preset", None)
    if preset_name in {"auto", "none"}:
        preset_name = None
    if preset_name:
        return _sanitize_name_fragment(preset_name)

    base_samples = getattr(args, "base_samples", "")
    if base_samples:
        return _sanitize_name_fragment(os.path.basename(os.path.normpath(base_samples)))

    return "unknown_model"


def _expected_message_len(args, payload_attr: str) -> int:
    lsb_payload_len = args.image_size * args.image_size * 3
    extra_payload_len = sum(getattr(args, payload_attr) or []) * 3
    return lsb_payload_len + extra_payload_len


def _format_bpp(message_len: int) -> str:
    bpp = message_len / _BPP_DENOMINATOR
    return str(int(bpp)) if float(bpp).is_integer() else format(bpp, ".12g")


def _build_experiment_dir_name(args, payload_attr: str, constraint_attr: str) -> str:
    model_category = _infer_model_category(args)
    constraint_height = int(getattr(args, constraint_attr))
    message_len = _expected_message_len(args, payload_attr)
    return f"{model_category}-{constraint_height}-{_format_bpp(message_len)}bpp"


def _create_batch_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch embed/extract runner for the hybrid STS method."
    )
    parser.add_argument("--mode", choices=["embed", "extract"], default="embed")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--base-samples", type=str, default="ref_imgs/face")
    parser.add_argument("--model-path", type=str, default="models/ffhq_10m.pt")
    parser.add_argument(
        "--model-preset",
        type=str,
        default="auto",
        choices=["auto", "none", "afhq_dog", "ffhq", "lsun_bedroom"],
    )
    parser.add_argument(
        "--sts-payload",
        type=int,
        nargs="+",
        default=[65536, 0, 0, 0, 0, 0, 0],
    )
    parser.add_argument("--sts-constraint-height", type=int, default=7)
    parser.add_argument("--sts-block-len", type=int, default=65536)
    parser.add_argument(
        "--use-marginal-probs",
        action="store_true",
        default=False,
        help="Ablation: use marginal P(s_b) instead of conditional P(s_b|ctx).",
    )
    parser.description = "Batch embed/extract runner for the hybrid STS method."
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to generate in embed mode.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Starting seed for embed mode.",
    )
    parser.add_argument(
        "--tasks-per-gpu",
        type=int,
        default=6,
        help="Number of producer processes to launch per GPU in embed mode.",
    )
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=10,
        help="Number of CPU consumer processes in embed mode.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0,3],
        help="Explicit GPU ids for embed mode. Defaults to all available GPUs.",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Existing output directory. In extract mode, messages are read from this directory.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output_hybrid_parallel",
        help="Root directory for auto-generated batch experiment folders.",
    )
    return parser


def _resolve_base_args(base_args: Optional[argparse.Namespace] = None) -> argparse.Namespace:
    bindings = _load_hybrid_bindings()
    args = bindings["create_argparser"]().parse_args([])
    if base_args is not None:
        for key, value in vars(base_args).items():
            if value is not None and hasattr(args, key):
                setattr(args, key, copy.deepcopy(value))
    bindings["resolve_model_preset"](args)
    return args


def deconstruct_image_to_bit_planes(image_np: np.ndarray) -> np.ndarray:
    h, w, _ = image_np.shape
    num_pixels = h * w
    bit_planes = np.zeros((24, num_pixels), dtype=int)
    for channel_idx in range(3):
        channel_flat = image_np[:, :, channel_idx].reshape(-1)
        for bit_idx in range(8):
            plane_idx = channel_idx * 8 + bit_idx
            bit_planes[plane_idx, :] = (channel_flat >> bit_idx) & 1
    return bit_planes


def _resolve_base_save_dir(
    args: argparse.Namespace,
    resume_dir: Optional[str],
    payload_attr: str,
    constraint_attr: str,
    output_root: str,
) -> Tuple[str, bool]:
    if resume_dir:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        return resume_dir, True

    experiment_dir_name = _build_experiment_dir_name(args, payload_attr, constraint_attr)
    base_save_dir = os.path.join(output_root, experiment_dir_name)
    return base_save_dir, os.path.isdir(base_save_dir)


def _collect_indexed_ids(directory: str, prefix: str, extension: str) -> set[int]:
    if not os.path.isdir(directory):
        return set()

    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.{re.escape(extension)}$")
    indexed_ids: set[int] = set()
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            indexed_ids.add(int(match.group(1)))
    return indexed_ids


def _compare_bitstrings(expected: str, actual: str) -> Tuple[int, int, float]:
    paired_correct = sum(int(lhs == rhs) for lhs, rhs in zip(expected, actual))
    total_bits = max(len(expected), len(actual))
    if total_bits == 0:
        return 0, 0, 1.0
    return paired_correct, total_bits, paired_correct / total_bits


def _extract_hybrid_message(stego_path: str, metadata_path: str) -> str:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    message_len = metadata["message_len"]
    constraint_height = metadata["sts_constraint_height"]
    seed = metadata["seed"]
    block_len = metadata["sts_block_len"]
    lsb_payload_bits = metadata["lsb_payload_bits_per_channel"]
    payloads_per_plane = metadata["payloads_per_plane"]
    sts_payload_len = metadata["sts_payload_len"]

    stego_image_np = np.array(Image.open(stego_path).convert("RGB"))
    num_pixels = stego_image_np.shape[0] * stego_image_np.shape[1]
    if num_pixels % block_len != 0:
        raise ValueError(f"Pixel count ({num_pixels}) not divisible by block length ({block_len}).")

    bit_planes = deconstruct_image_to_bit_planes(stego_image_np)

    extracted_lsb_bits: list[int] = []
    for plane_idx in [0, 8, 16]:
        extracted_lsb_bits.extend(bit_planes[plane_idx][:lsb_payload_bits].tolist())

    num_blocks = num_pixels // block_len
    samplers_cache: Dict[Tuple[int, int, int, int], STSSampler] = {}
    extracted_sts_bits: list[int] = []

    filling_order = []
    for bit_in_channel in range(1, 8):
        for channel_idx in range(3):
            filling_order.append(channel_idx * 8 + bit_in_channel)

    for plane_idx in filling_order:
        payload_for_plane = payloads_per_plane[plane_idx]
        if payload_for_plane == 0:
            continue

        stego_plane_bits = bit_planes[plane_idx]
        payloads_per_block = [payload_for_plane // num_blocks] * num_blocks
        for block_idx in range(payload_for_plane % num_blocks):
            payloads_per_block[block_idx] += 1

        for block_idx, payload_size in enumerate(payloads_per_block):
            if payload_size == 0:
                continue

            start_idx = block_idx * block_len
            end_idx = (block_idx + 1) * block_len
            block_bits = stego_plane_bits[start_idx:end_idx]

            sampler_key = (payload_size, constraint_height, block_len, seed)
            if sampler_key not in samplers_cache:
                samplers_cache[sampler_key] = STSSampler(
                    c=payload_size,
                    h=constraint_height,
                    n=block_len,
                    matrix_seed=seed,
                    sample_seed=seed + plane_idx * num_blocks + block_idx,
                )

            extracted_sts_bits.extend(samplers_cache[sampler_key].matvec(block_bits).tolist())

    full_extracted_bits = np.concatenate(
        [
            np.asarray(extracted_lsb_bits, dtype=int),
            np.asarray(extracted_sts_bits[:sts_payload_len], dtype=int),
        ]
    )
    return "".join(map(str, full_extracted_bits.astype(int)))[:message_len]

# --- Producer-Consumer Functions for Dataset Generation ---

def hybrid_producer(task_queue: multiprocessing.Queue, seed_list: List[int], gpu_id: int, base_args, temp_dir: str):
    """
    Producer: Generates probability data using GPU and saves it to a temporary file.
    This is identical to the producer in run_batch.py.
    """
    # --- Suppress all output from this worker ---
    sys.stdout = open(os.devnull, 'w')
    logger.configure(dir=temp_dir, format_strs=['log']) # Only log to file

    bindings = _load_hybrid_bindings()
    prepare_data_for_pair = bindings["prepare_data_for_pair"]
    resolve_model_preset = bindings["resolve_model_preset"]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    args = copy.deepcopy(base_args)
    resolve_model_preset(args)

    MAX_TEMP_FILES = 30 # Define the maximum number of temporary files

    for seed in seed_list:
        # Check temporary file count before proceeding
        while True:
            current_temp_files = glob.glob(os.path.join(temp_dir, '*.npz'))
            if len(current_temp_files) >= MAX_TEMP_FILES:
                # sys.stderr.write(f"Producer (GPU {gpu_id}) pausing: {len(current_temp_files)} temp files >= {MAX_TEMP_FILES}. Waiting...\n")
                time.sleep(60) # Wait for 5 seconds before checking again
            else:
                break # Continue if below the limit

        args.seed = seed
        try:
            all_data = prepare_data_for_pair(args)
            temp_data_path = os.path.join(temp_dir, f"data_{seed}.npz")
            np.savez_compressed(
                temp_data_path,
                probs_maps=all_data[0],
                all_marginal_probs=all_data[1],
                template_cover_np=all_data[2],
                cover_conditional_probs_np=all_data[3]
            )
            task_queue.put({'task_id': seed, 'seed': seed, 'temp_data_path': temp_data_path})
        except Exception as e:
            sys.stderr.write(f"Producer error for task {seed}: {e}\n")
            task_queue.put({'task_id': seed, 'seed': seed, 'error': str(e)})

def hybrid_consumer(task_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue, base_save_dir: str, base_args, consumer_id: int):
    """
    Consumer: Uses CPU to perform the HYBRID embedding logic.
    """
    # --- Suppress all output from this worker ---
    sys.stdout = open(os.devnull, 'w')
    log_dir = os.path.join(base_save_dir, 'logs_consumer')
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir, format_strs=['log']) # Only log to file

    bindings = _load_hybrid_bindings()
    embed_using_precalculated_data_hybrid = bindings["embed_using_precalculated_data_hybrid"]
    resolve_model_preset = bindings["resolve_model_preset"]

    rng = np.random.default_rng(seed=(base_args.seed + consumer_id))
    from PIL import Image

    cover_dir, stega_dir, msg_dir, meta_dir = (os.path.join(base_save_dir, d) for d in ['cover', 'stega', 'message', 'metadata'])

    while True:
        task_data = task_queue.get()
        if task_data is None: break

        task_id = task_data['task_id']
        if 'error' in task_data:
            done_queue.put({'task_id': task_id, 'status': 'failed_producer'})
            continue

        temp_data_path = task_data.get('temp_data_path')
        if not temp_data_path or not os.path.exists(temp_data_path):
            done_queue.put({'task_id': task_id, 'status': 'failed_consumer_no_temp_file'})
            continue
        
        try:
            args = copy.deepcopy(base_args)
            args.seed = task_data['seed']
            args.save_dir = base_save_dir
            resolve_model_preset(args)

            # Load all pre-calculated data from the temporary file
            with np.load(temp_data_path) as data:
                probs_maps = data['probs_maps']
                cover_img_np = data['template_cover_np']
            
            # The temp file is no longer needed after loading
            os.remove(temp_data_path) 

            # Define file paths
            file_suffix = f"{task_id}"
            cover_path = os.path.join(cover_dir, f"cover_{file_suffix}.png")
            stega_path = os.path.join(stega_dir, f"stega_{file_suffix}.png")
            message_path = os.path.join(msg_dir, f"message_{file_suffix}.txt")
            metadata_path = os.path.join(meta_dir, f"metadata_{file_suffix}.json")
            
            # --- 1. Save Cover Image ---
            Image.fromarray(cover_img_np, 'RGB').save(cover_path)

            # --- 2. Generate and Prepare Message for Embedding ---
            # The LSB part will be generated via sampling inside the embedder.
            # We only need to generate the STS part and a placeholder for the LSB part.
            lsb_payload_len = args.image_size * args.image_size * 3
            if args.sts_payload:
                sts_payload_len = sum(args.sts_payload) * 3
            else:
                sts_payload_len = 0
            
            payload_size = lsb_payload_len + sts_payload_len
            
            message_with_placeholder = "".join(map(str, rng.integers(0, 2, payload_size)))

            # --- 3. Embed message using the PRE-CALCULATED data ---
            # This function returns the *actual* embedded message, including the sampled LSB part.
            stego_image_np, metadata, final_embedded_message_np = embed_using_precalculated_data_hybrid(
                args,
                message_with_placeholder,
                probs_maps,
                cover_img_np
            )

            # --- 4. Save the actual stego data ---
            # Save the *actual* message that was embedded
            final_embedded_message_str = "".join(map(str, final_embedded_message_np.astype(int)))
            with open(message_path, 'w') as f:
                f.write(final_embedded_message_str)

            # Save stego image and metadata
            Image.fromarray(stego_image_np, 'RGB').save(stega_path)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            done_queue.put({'task_id': task_id, 'status': 'success'})

        except Exception as e:
            sys.stderr.write(f"Consumer error for task {task_id}: {e}\n")
            done_queue.put({'task_id': task_id, 'status': 'failed_consumer'})
            if temp_data_path and os.path.exists(temp_data_path):
                try: os.remove(temp_data_path)
                except OSError: pass

def run_hybrid_experiment(
    num_images: int,
    seed_start: int = 1,
    tasks_per_gpu: int = 4,
    cpu_cores: int = None,
    gpus: List[int] = None,
    resume_dir: str = None,
    sts_payload: List[int] = None,
    base_args: Optional[argparse.Namespace] = None,
    output_root: str = "output_hybrid_parallel",
):
    """
    Main runner for the hybrid dataset generation experiment.
    """
    base_save_dir = ""
    temp_dir = ""
    base_args = _resolve_base_args(base_args)
    if sts_payload is not None:
        base_args.sts_payload = sts_payload
    try:
        base_save_dir, using_existing_dir = _resolve_base_save_dir(
            base_args,
            resume_dir,
            "sts_payload",
            "sts_constraint_height",
            output_root,
        )
        if resume_dir:
            print(f"--- Resuming HYBRID dataset generation in: {base_save_dir} ---")
        else:
            os.makedirs(base_save_dir, exist_ok=True)
            if using_existing_dir:
                print(f"--- Reusing HYBRID dataset generation directory: {base_save_dir} ---")
            else:
                print(f"--- Running HYBRID dataset generation in: {base_save_dir} ---")

        temp_dir = os.path.join(base_save_dir, 'temp_probs')
        os.makedirs(temp_dir, exist_ok=True)
        for sub_dir in ['cover', 'stega', 'message', 'metadata']:
            os.makedirs(os.path.join(base_save_dir, sub_dir), exist_ok=True)

        logger.configure(dir=base_save_dir, format_strs=['stdout', 'log'])

        try:
            import torch
            num_gpus_available = torch.cuda.device_count()
            if num_gpus_available == 0: raise EnvironmentError("This mode requires at least one NVIDIA GPU.")
        except (ImportError, Exception):
            raise EnvironmentError("PyTorch or CUDA not available.")

        target_seeds = set(range(seed_start, seed_start + num_images))
        if resume_dir or using_existing_dir:
            existing_seeds = {int(match.group(1)) for f in os.listdir(os.path.join(base_save_dir, 'stega')) if (match := re.match(r'stega_(\d+)\.png', f))}
            seeds_to_generate = sorted(list(target_seeds - existing_seeds))
            print(f"Found {len(existing_seeds)} existing images. Need to generate {len(seeds_to_generate)} more.")
        else:
            seeds_to_generate = sorted(list(target_seeds))
        
        if not seeds_to_generate:
            print("All target images already exist. Nothing to do.")
            return

        ctx = multiprocessing.get_context('spawn')
        manager = ctx.Manager()
        task_queue, done_queue = manager.Queue(), manager.Queue()
        
        gpu_ids = gpus if gpus is not None else list(range(num_gpus_available))
        num_producers = len(gpu_ids) * tasks_per_gpu
        tasks_for_producers = np.array_split(seeds_to_generate, num_producers)
        
        producers = []
        producer_id_counter = 0
        print(f"Starting {num_producers} producer processes on GPUs: {gpu_ids}.")
        for gpu_id in gpu_ids:
            for _ in range(tasks_per_gpu):
                if producer_id_counter < len(tasks_for_producers):
                    seed_list = tasks_for_producers[producer_id_counter]
                    if len(seed_list) > 0:
                        p = ctx.Process(target=hybrid_producer, args=(task_queue, list(seed_list), gpu_id, base_args, temp_dir))
                        producers.append(p)
                        p.start()
                    producer_id_counter += 1
        
        num_consumers = cpu_cores if cpu_cores and cpu_cores > 0 else (os.cpu_count() or 1)
        print(f"Starting {num_consumers} CPU Consumer processes.")
        consumer_pool = MyPool(processes=num_consumers)
        for i in range(num_consumers):
            consumer_pool.apply_async(hybrid_consumer, args=(task_queue, done_queue, base_save_dir, base_args, i))

        pbar = tqdm(total=len(seeds_to_generate), desc="Hybrid Dataset Generation")
        completed_count = 0
        while completed_count < len(seeds_to_generate):
            done_queue.get()
            completed_count += 1
            pbar.update(1)
        pbar.close()

        for _ in range(num_consumers): task_queue.put(None)
        consumer_pool.close()
        consumer_pool.join()
        for p in producers: p.join()

        print("\n--- Hybrid Dataset Generation Finished ---")
        print(f"All output saved in: {os.path.abspath(base_save_dir)}")
    
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_hybrid_extraction(
    resume_dir: Optional[str] = None,
    sts_payload: Optional[List[int]] = None,
    base_args: Optional[argparse.Namespace] = None,
    output_root: str = "output_hybrid_parallel",
) -> Dict[str, Any]:
    base_args = copy.deepcopy(base_args) if base_args is not None else _create_batch_argparser().parse_args([])
    if sts_payload is not None:
        base_args.sts_payload = sts_payload

    base_save_dir, _ = _resolve_base_save_dir(
        base_args,
        resume_dir,
        "sts_payload",
        "sts_constraint_height",
        output_root,
    )
    if not os.path.isdir(base_save_dir):
        raise FileNotFoundError(f"Extraction directory not found: {base_save_dir}")

    logger.configure(dir=base_save_dir, format_strs=["stdout", "log"])

    stega_dir = os.path.join(base_save_dir, "stega")
    message_dir = os.path.join(base_save_dir, "message")
    metadata_dir = os.path.join(base_save_dir, "metadata")
    message_ext_dir = os.path.join(base_save_dir, "message_ext")
    extract_status_dir = os.path.join(base_save_dir, "extract_status")
    os.makedirs(message_ext_dir, exist_ok=True)
    os.makedirs(extract_status_dir, exist_ok=True)

    stega_ids = _collect_indexed_ids(stega_dir, "stega", "png")
    message_ids = _collect_indexed_ids(message_dir, "message", "txt")
    metadata_ids = _collect_indexed_ids(metadata_dir, "metadata", "json")
    task_ids = sorted(stega_ids & message_ids & metadata_ids)

    if not task_ids:
        raise RuntimeError(
            f"No complete stego/message/metadata triplets found under {os.path.abspath(base_save_dir)}"
        )

    summary: Dict[str, Any] = {
        "mode": "extract",
        "method": "hybrid",
        "base_save_dir": os.path.abspath(base_save_dir),
        "processed": 0,
        "failures": 0,
        "exact_matches": 0,
        "exact_match_rate": 0.0,
        "correct_bits": 0,
        "total_bits": 0,
        "bit_accuracy": 0.0,
    }

    for task_id in tqdm(task_ids, desc="Hybrid Extraction"):
        stego_path = os.path.join(stega_dir, f"stega_{task_id}.png")
        message_path = os.path.join(message_dir, f"message_{task_id}.txt")
        metadata_path = os.path.join(metadata_dir, f"metadata_{task_id}.json")
        extracted_path = os.path.join(message_ext_dir, f"message_{task_id}.txt")
        status_path = os.path.join(extract_status_dir, f"extract_{task_id}.json")

        try:
            extracted_message = _extract_hybrid_message(stego_path, metadata_path)
            with open(extracted_path, "w") as f:
                f.write(extracted_message)

            with open(message_path, "r") as f:
                original_message = f.read().strip()

            correct_bits, total_bits, bit_accuracy = _compare_bitstrings(
                original_message,
                extracted_message,
            )
            exact_match = original_message == extracted_message

            status: Dict[str, Any] = {
                "task_id": task_id,
                "success": True,
                "exact_match": exact_match,
                "correct_bits": correct_bits,
                "total_bits": total_bits,
                "bit_accuracy": bit_accuracy,
                "original_length": len(original_message),
                "extracted_length": len(extracted_message),
                "message_path": message_path,
                "message_ext_path": extracted_path,
                "metadata_path": metadata_path,
                "stego_path": stego_path,
            }

            summary["processed"] += 1
            summary["exact_matches"] += int(exact_match)
            summary["correct_bits"] += correct_bits
            summary["total_bits"] += total_bits
        except Exception as exc:
            status = {
                "task_id": task_id,
                "success": False,
                "exact_match": False,
                "correct_bits": 0,
                "total_bits": 0,
                "bit_accuracy": 0.0,
                "message_path": message_path,
                "message_ext_path": extracted_path,
                "metadata_path": metadata_path,
                "stego_path": stego_path,
                "error": str(exc),
            }
            summary["failures"] += 1

        with open(status_path, "w") as f:
            json.dump(status, f, indent=4)

    if summary["processed"] > 0:
        summary["exact_match_rate"] = summary["exact_matches"] / summary["processed"]
    if summary["total_bits"] > 0:
        summary["bit_accuracy"] = summary["correct_bits"] / summary["total_bits"]

    summary_path = os.path.join(base_save_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n--- Hybrid Extraction Finished ---")
    print(f"Processed samples: {summary['processed']}")
    print(f"Failed samples: {summary['failures']}")
    print(f"Exact matches: {summary['exact_matches']}/{summary['processed']}")
    print(f"Extraction accuracy: {summary['bit_accuracy']:.6f}")
    print(f"Summary saved to: {os.path.abspath(summary_path)}")
    return summary


def main() -> None:
    args = _create_batch_argparser().parse_args()

    if args.mode == "embed":
        if args.num_images <= 0:
            raise ValueError("--num-images must be a positive integer in embed mode.")
        run_hybrid_experiment(
            num_images=args.num_images,
            seed_start=args.seed_start,
            tasks_per_gpu=args.tasks_per_gpu,
            cpu_cores=args.cpu_cores,
            gpus=args.gpus,
            resume_dir=args.resume_dir,
            sts_payload=args.sts_payload,
            base_args=args,
            output_root=args.output_root,
        )
        return

    run_hybrid_extraction(
        resume_dir=args.resume_dir,
        sts_payload=args.sts_payload,
        base_args=args,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
