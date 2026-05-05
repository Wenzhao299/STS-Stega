"""
stega_hybrid.py

This script implements a hybrid steganography method for demonstration.
- The three LSB planes (R, G, B) are embedded using direct LSB replacement.
- Higher bit planes (1-7 for each channel) are embedded using STScp 
  conditional probability sampling, with the LSB planes as context.
"""

import os
import math
import argparse
import time
import random
from typing import Optional, Union, Tuple, List, Dict, Any
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# --- Global Logger Replacement ---
import sys
import null_logger
# sys.modules['guided_diffusion.logger'] = null_logger
# --- End of Replacement ---

from utils import ImageSettings, get_probs_indices_from_diffu, load_reference
from resizer import Resizer
from guided_diffusion import logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from stscp_sampler_new import STSSampler
import json
import multiprocessing

from stega_stscp import (
    prepare_data_for_pair,
    deconstruct_image_to_bit_planes,
    reconstruct_image_from_bit_planes,
    calculate_conditional_entropies,
    calculate_conditional_probs_for_plane
)


def get_all_marginal_probs(probs_maps: np.ndarray) -> np.ndarray:
    """
    Computes the marginal probability P(s_i=b) for each bit position i, for all pixels.
    This creates the probability distributions for all 24 bit-planes.
    
    Args:
        probs_maps (np.ndarray): A (3, H*W, 256) array of channel probabilities.
    Returns:
        np.ndarray: A (H*W, 24, 2) array where result[px_idx, i, b] = P(s_i=b).
    """
    num_pixels = probs_maps.shape[1]
    n_bits_per_pixel = 24
    all_marginal_probs = np.zeros((num_pixels, n_bits_per_pixel, 2))

    val_range = np.arange(256)
    for i in range(n_bits_per_pixel):
        if i < 8: channel_idx, bit_in_channel = 0, i
        elif i < 16: channel_idx, bit_in_channel = 1, i - 8
        else: channel_idx, bit_in_channel = 2, i - 16

        mask = 1 << bit_in_channel
        p0_mask = (val_range & mask) == 0
        p1_mask = ~p0_mask
        
        p0 = np.sum(probs_maps[channel_idx, :, :] * p0_mask, axis=1)
        p1 = np.sum(probs_maps[channel_idx, :, :] * p1_mask, axis=1)
        
        all_marginal_probs[:, i, 0] = p0
        all_marginal_probs[:, i, 1] = p1

    return all_marginal_probs

def calculate_conditional_entropies(conditional_probs: np.ndarray) -> np.ndarray:
    """
    Calculates the total Shannon entropy for each bit-plane based on pre-calculated
    conditional probabilities P(s_i | context).
    """
    probs = conditional_probs + 1e-9
    entropies_per_bit = -np.sum(probs * np.log2(probs), axis=2)
    total_plane_entropies = np.sum(entropies_per_bit, axis=0)
    return total_plane_entropies

def calculate_conditional_probs_for_plane(
    plane_idx: int,
    probs_maps: np.ndarray,
    current_stego_planes: np.ndarray
) -> np.ndarray:
    """
    Calculates the conditional probability P(s_i=b | context) for a given bit-plane.
    The context is the set of already determined lower-order bits in the same channel.
    """
    num_pixels = probs_maps.shape[1]
    
    if plane_idx < 8: channel_idx, bit_in_channel = 0, plane_idx
    elif plane_idx < 16: channel_idx, bit_in_channel = 1, plane_idx - 8
    else: channel_idx, bit_in_channel = 2, plane_idx - 16
    
    channel_probs_map = probs_maps[channel_idx]
    
    value_range = np.arange(256, dtype=np.uint8)[None, :]
    context_match_mask = np.ones((num_pixels, 256), dtype=bool)

    for b in range(bit_in_channel):
        lower_plane_idx = channel_idx * 8 + b
        lower_plane_bits = current_stego_planes[lower_plane_idx, :][:, None]
        plane_b_match = ((value_range >> b) & 1) == lower_plane_bits
        context_match_mask &= plane_b_match

    p_context = np.sum(channel_probs_map * context_match_mask, axis=1, keepdims=True)
    p_context[p_context < 1e-9] = 1.0

    b_is_1_mask = ((value_range >> bit_in_channel) & 1) == 1
    p_b1_and_context = np.sum(channel_probs_map * context_match_mask * b_is_1_mask, axis=1, keepdims=True)
    
    p_b1_cond = np.clip((p_b1_and_context / p_context).squeeze(), 0, 1)
    p_b0_cond = 1.0 - p_b1_cond

    return np.stack([p_b0_cond, p_b1_cond], axis=1)

def deconstruct_image_to_bit_planes(image_np: np.ndarray) -> np.ndarray:
    h, w, _ = image_np.shape
    num_pixels = h * w
    bit_planes = np.zeros((24, num_pixels), dtype=int)
    for channel_idx in range(3):
        channel_flat = image_np[:, :, channel_idx].flatten()
        for bit_idx in range(8):
            plane_idx = channel_idx * 8 + bit_idx
            bit_planes[plane_idx, :] = (channel_flat >> bit_idx) & 1
    return bit_planes

def reconstruct_image_from_bit_planes(modified_planes: np.ndarray, h: int, w: int) -> np.ndarray:
    num_pixels = h * w
    image_np = np.zeros((h, w, 3), dtype=np.uint8)
    for channel_idx in range(3):
        val = np.zeros(num_pixels, dtype=np.int32)
        for bit_idx in range(8):
            plane_idx = channel_idx * 8 + bit_idx
            val += modified_planes[plane_idx].astype(np.int32) << bit_idx
        image_np[:, :, channel_idx] = val.reshape((h, w))
    return image_np

def generate_cover_from_probs(probs_maps: np.ndarray, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    cdfs = np.cumsum(probs_maps, axis=2)
    uniform_samples = rng.random((3, h * w, 1))
    sampled_indices = np.argmax(uniform_samples < cdfs, axis=2).astype(np.uint8)
    cover_image_np = np.zeros((h, w, 3), dtype=np.uint8)
    cover_image_np[:, :, 0] = sampled_indices[0, :].reshape((h, w))
    cover_image_np[:, :, 1] = sampled_indices[1, :].reshape((h, w))
    cover_image_np[:, :, 2] = sampled_indices[2, :].reshape((h, w))
    return cover_image_np

@torch.no_grad()
def run_diffusion(args):
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_fp16: model.convert_to_fp16()
    model.eval()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    resizers = (Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device), Resizer(shape_d, args.down_N).to(next(model.parameters()).device))
    
    data_loader = load_reference(args.base_samples, args.batch_size, image_size=args.image_size, class_cond=args.class_cond)
    model_kwargs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in next(data_loader).items()}
    
    output = diffusion.p_sample_loop(
        model, (args.batch_size, 3, args.image_size, args.image_size),
        index=1, clip_denoised=args.clip_denoised, model_kwargs=model_kwargs,
        resizers=resizers, range_t=args.range_t, seed=args.seed
    )
    return output.get('sample'), output.get('mu'), output.get('sigma')

def generate_probabilities(args):
    _, mu_norm, sigma_norm = run_diffusion(args)
    if mu_norm is None or sigma_norm is None: raise RuntimeError("Diffusion model did not return mu and sigma.")
    _, probs_maps_list = get_probs_indices_from_diffu(mu_norm.cpu().numpy().flatten(), sigma_norm.cpu().numpy().flatten(), 3, args.image_size, args.image_size)
    return np.array(probs_maps_list), get_all_marginal_probs(np.array(probs_maps_list))

def calculate_cover_channel_conditional_probs_worker(args_tuple):
    (channel_idx, probs_maps, template_cover_bits_np, image_size) = args_tuple
    num_pixels = image_size * image_size
    cover_cond_probs_channel_np = np.zeros((num_pixels, 8, 2))
    pbar_channel = range(8)
    for bit_in_channel in pbar_channel:
        plane_idx = channel_idx * 8 + bit_in_channel
        cond_probs = calculate_conditional_probs_for_plane(plane_idx, probs_maps, template_cover_bits_np)
        cover_cond_probs_channel_np[:, bit_in_channel, :] = cond_probs
    return channel_idx, cover_cond_probs_channel_np

def prepare_data_for_pair(args):
    h, w, num_pixels = args.image_size, args.image_size, args.image_size * args.image_size
    probs_maps, all_marginal_probs = generate_probabilities(args)
    rng = np.random.default_rng(args.seed)
    template_cover_np = generate_cover_from_probs(probs_maps, h, w, rng)
    template_cover_bits_np = deconstruct_image_to_bit_planes(template_cover_np)
    
    cover_prob_worker_args = [(i, probs_maps, template_cover_bits_np, args.image_size) for i in range(3)]
    cover_conditional_probs_np = np.zeros((num_pixels, 24, 2))

    try:
        with multiprocessing.Pool(processes=min(3, (os.cpu_count() or 1))) as pool:
            results = list(pool.imap_unordered(calculate_cover_channel_conditional_probs_worker, cover_prob_worker_args))
    except Exception:
        results = [calculate_cover_channel_conditional_probs_worker(arg) for arg in cover_prob_worker_args]

    for channel_idx, channel_cond_probs in results:
        for bit_in_channel in range(8):
            plane_idx = channel_idx * 8 + bit_in_channel
            cover_conditional_probs_np[:, plane_idx, :] = channel_cond_probs[:, bit_in_channel, :]
    
    return probs_maps, all_marginal_probs, template_cover_np, cover_conditional_probs_np

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        attention_resolutions="16", 
        diffusion_steps=1000, 
        image_size=256, 
        learn_sigma=True,
        noise_schedule="linear", 
        num_channels=128, 
        num_head_channels=64, 
        num_res_blocks=1,
        resblock_updown=True, 
        timestep_respacing="100", 
        clip_denoised=True, 
        num_samples=1, 
        batch_size=1, 
        down_N=16, 
        range_t=20, 
        use_ddim=False, 
        base_samples="ref_imgs/face",
        model_path="models/ffhq_10m.pt", 
        save_dir="output_stscp_new", 
        seed=1, 
        message_input="message.txt",
        mode="embed", 
        sts_payload=[32768, 0, 0, 0, 0, 0, 0], # New: Direct payload specification
        sts_constraint_height=7, 
        sts_block_len=65536,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    # --lsb_payload_bits is removed as we use STS for all planes
    # parser.add_argument('--sts_payload', type=int, nargs='+', help='Directly specify payload for each non-LSB plane layer [1-7]. e.g., --sts_payload 1000 500')
    for action in parser._actions:
        if action.dest == 'mode':
            action.choices = ['embed', 'extract']
            break
    return parser

# ==============================================================================
# == STS EMBEDDING LOGIC and EXTRACT LOGIC
# ==============================================================================

def embed_channel_sts_worker(args_tuple: tuple) -> Tuple[int, Dict[int, np.ndarray], Any]:
    """
    Worker function for the STS method. This handles STScp embedding for
    all planes (0-7) for a single channel.
    """
    # Unpack arguments
    (
        channel_idx, 
        args, 
        probs_maps, 
        initial_stego_bits,
        sts_payloads_per_plane, 
        sts_plane_to_chunk_map,
        samplers_cache
    ) = args_tuple

    h, w = args.image_size, args.image_size
    num_pixels = h * w
    block_len = args.sts_block_len

    # The worker operates on a local copy of the bit planes.
    stego_bits_for_worker = np.copy(initial_stego_bits)
    
    modified_bits_dict = {}

    # Process all planes from bit 0 up to 7 for the assigned channel
    for bit_in_channel in range(8):
        plane_idx = channel_idx * 8 + bit_in_channel
        payload_for_plane = sts_payloads_per_plane[plane_idx]

        cond_probs = calculate_conditional_probs_for_plane(
            plane_idx, probs_maps, stego_bits_for_worker
        )

        modified_plane = None

        if payload_for_plane == 0:
            # WET RUN: Sample bits based on conditional probability
            worker_rng_seed = args.seed + 1000 * channel_idx + 100 * bit_in_channel
            rng = np.random.default_rng(worker_rng_seed)
            uniform_samples = rng.random(num_pixels)
            modified_plane = (uniform_samples > cond_probs[:, 0]).astype(int)
        
        else: # DRY RUN: Embed payload
            message_chunk = sts_plane_to_chunk_map.get(plane_idx, np.array([], dtype=int))
            num_blocks = num_pixels // block_len
            payloads_per_block = [payload_for_plane // num_blocks] * num_blocks
            for i in range(payload_for_plane % num_blocks):
                payloads_per_block[i] += 1
            
            modified_plane_temp = np.zeros(num_pixels, dtype=int)
            msg_offset_in_plane = 0
            embedding_successful = True

            for i in range(num_blocks):
                start_idx, end_idx = i * block_len, (i + 1) * block_len
                block_probs = cond_probs[start_idx:end_idx]
                payload_size = payloads_per_block[i]
                block_message_chunk = message_chunk[msg_offset_in_plane : msg_offset_in_plane + payload_size]
                msg_offset_in_plane += payload_size
                
                # --- OPTIMIZATION: Use Sampler Cache ---
                sampler_key = (payload_size, args.sts_constraint_height, block_len, args.seed)
                if sampler_key not in samplers_cache:
                    matrix_seed = args.seed # Use base seed for matrix reproducibility
                    sample_seed = args.seed + plane_idx * num_blocks + i # Unique seed for sampling
                    samplers_cache[sampler_key] = STSSampler(
                        c=payload_size, 
                        h=args.sts_constraint_height, 
                        n=block_len, 
                        matrix_seed=matrix_seed, 
                        sample_seed=sample_seed
                    )
                sampler = samplers_cache[sampler_key]
                # --- END OPTIMIZATION ---

                modified_block, _ = sampler.sample_bit_plane(block_probs, block_message_chunk, calculate_posterior=False, verbose=False)

                if modified_block is None:
                    embedding_successful = False
                    break 
                modified_plane_temp[start_idx:end_idx] = modified_block

            if not embedding_successful:
                return channel_idx, None, plane_idx
            
            modified_plane = modified_plane_temp

        modified_bits_dict[plane_idx] = modified_plane
        stego_bits_for_worker[plane_idx, :] = modified_plane

    return channel_idx, modified_bits_dict, None

def embed_using_precalculated_data_sts(args: argparse.Namespace, message_str: str, probs_maps: np.ndarray, template_cover_np: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Performs the core STS embedding logic using pre-calculated data.
    This function does NOT run the diffusion model.
    """
    h, w = args.image_size, args.image_size
    
    sts_msg_part = np.array([int(b) for b in message_str], dtype=int)
    
    # The working bits are initialized from the randomly sampled cover.
    working_stego_bits = deconstruct_image_to_bit_planes(template_cover_np)

    # --- New logic: Directly construct payload distribution from args ---
    payloads_per_plane = np.zeros(24, dtype=int)
    if args.sts_payload:
        for bit_level_from_1 in range(1, 8): # For planes 1 to 7
            payload_idx = bit_level_from_1 - 1
            if payload_idx < len(args.sts_payload):
                payload_for_level = args.sts_payload[payload_idx]
                for channel_idx in range(3): # For R, G, B channels
                    plane_idx = channel_idx * 8 + bit_level_from_1
                    payloads_per_plane[plane_idx] = payload_for_level
    
    total_payload_specified = np.sum(payloads_per_plane)
    if len(sts_msg_part) > total_payload_specified:
        print(f"Warning: message length ({len(sts_msg_part)}) exceeds specified payload ({total_payload_specified}). Truncating message.")
        sts_msg_part = sts_msg_part[:total_payload_specified]
    elif len(sts_msg_part) < total_payload_specified:
        print(f"Warning: message length ({len(sts_msg_part)}) is less than specified payload ({total_payload_specified}). This may lead to incorrect extraction if not intended.")

    # Map message chunks to the planes that will carry a payload
    message_offset = 0
    plane_to_chunk_map = {}
    
    # The embedding order is sequential by plane index: R0, R1... R7, G0, G1...
    filling_order = list(range(24))

    for plane_idx in filling_order:
        payload = payloads_per_plane[plane_idx]
        if payload > 0:
            plane_to_chunk_map[plane_idx] = sts_msg_part[message_offset : message_offset + payload]
            message_offset += payload

    # Prepare arguments for parallel workers
    worker_args_list = []
    samplers_cache = {} # Create one cache to be shared by all channel workers
    for channel_idx in range(3):
        worker_args_list.append((
            channel_idx, args, probs_maps, working_stego_bits,
            payloads_per_plane, plane_to_chunk_map,
            samplers_cache # Pass the shared cache
        ))

    num_procs = min(3, os.cpu_count() or 1)
    final_stego_bits_np = np.copy(working_stego_bits)
    
    pool = multiprocessing.Pool(processes=num_procs)
    try:
        # Using imap_unordered to process as results become available
        results = list(pool.imap_unordered(embed_channel_sts_worker, worker_args_list))
    finally:
        pool.close()
        pool.join()

    # Consolidate results from all workers
    for result in results:
        _, modified_bits_dict, _ = result
        if modified_bits_dict:
            for plane_idx, bits in modified_bits_dict.items():
                final_stego_bits_np[plane_idx, :] = bits

    stego_image_np = reconstruct_image_from_bit_planes(final_stego_bits_np, h, w)
    
    metadata = {
        'message_len': len(sts_msg_part),
        'sts_payload': args.sts_payload,
        'sts_payload_len': len(sts_msg_part),
        'sts_constraint_height': int(args.sts_constraint_height),
        'sts_block_len': int(args.sts_block_len),
        'seed': int(args.seed),
        'payloads_per_plane': [int(p) for p in payloads_per_plane],
        'algorithm': 'sts' # New algorithm name
    }

    return stego_image_np, metadata

def embed_message_sts(args: argparse.Namespace, message_str: str) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Main function for the STS embedding process (for single runs).
    It generates data and then calls the core embedding logic.
    """
    logger.log("--- Starting STS Embedding ---")
    
    # 1. Generate base probabilities and template cover image
    data_tuple = prepare_data_for_pair(args)
    probs_maps = data_tuple[0]
    template_cover_np = data_tuple[2]
    logger.log("Generated base probabilities and template cover image.")
    
    # 2. Call the core embedding logic with the pre-calculated data
    stego_image_np, metadata = embed_using_precalculated_data_sts(
        args, message_str, probs_maps, template_cover_np
    )

    return stego_image_np, metadata, template_cover_np

def extract_message_sts(args):
    logger.log("--- Starting STS Extraction ---")

    # Construct paths
    stego_path = os.path.join(args.save_dir, "stego.png")
    metadata_path = os.path.join(args.save_dir, "metadata.json")
    output_path = os.path.join(args.save_dir, "message_extracted.txt")

    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if metadata.get('algorithm') != 'sts':
            print("Algorithm mismatch in metadata. Expected 'sts'.")
            return

        message_len = metadata['message_len']
        constraint_height = metadata['sts_constraint_height']
        seed = metadata['seed']
        block_len = metadata.get('sts_block_len')
        payloads_per_plane = metadata['payloads_per_plane']
        
    except FileNotFoundError:
        logger.error(f"Metadata file '{metadata_path}' not found.")
        return
    except KeyError as e:
        logger.error(f"Metadata is missing a required key: {e}")
        return

    # Load stego image
    stego_image_np = np.array(Image.open(stego_path).convert('RGB'))
    h, w, num_pixels = stego_image_np.shape[0], stego_image_np.shape[1], stego_image_np.shape[0] * stego_image_np.shape[1]
    
    if num_pixels % block_len != 0:
        raise ValueError(f"Pixel count ({num_pixels}) not divisible by block length ({block_len}).")

    bit_planes = deconstruct_image_to_bit_planes(stego_image_np)
    
    extracted_bits = []
    num_blocks = num_pixels // block_len
    samplers_cache = {} # Cache for extractor samplers
    
    # The extraction order must match the embedding order: R0..R7, G0..G7, B0..B7
    extraction_order = list(range(24))

    for plane_idx in extraction_order:
        payload_for_plane = payloads_per_plane[plane_idx]
        if payload_for_plane == 0:
            continue

        stego_plane_bits = bit_planes[plane_idx]
        
        payloads_per_block = [payload_for_plane // num_blocks] * num_blocks
        for i in range(payload_for_plane % num_blocks):
            payloads_per_block[i] += 1
        
        for i in range(num_blocks):
            payload_size = payloads_per_block[i]
            if payload_size == 0:
                continue
            stego_block = stego_plane_bits[i*block_len:(i+1)*block_len]
            
            # --- OPTIMIZATION: Use Sampler Cache and matvec ---
            sampler_key = (payload_size, constraint_height, block_len, seed)
            if sampler_key not in samplers_cache:
                matrix_seed = seed # Use base seed for matrix reproducibility
                sample_seed = seed + plane_idx * num_blocks + i # Matching sample seed
                samplers_cache[sampler_key] = STSSampler(
                    c=payload_size, 
                    h=constraint_height, 
                    n=block_len, 
                    matrix_seed=matrix_seed, 
                    sample_seed=sample_seed
                )
            sampler = samplers_cache[sampler_key]
            extracted_bits.extend(sampler.matvec(stego_block).tolist())
            # --- END OPTIMIZATION ---
            
    extracted_str = "".join(map(str, extracted_bits))[:message_len]
    with open(output_path, 'w') as f: f.write(extracted_str)
    logger.log(f"Extracted {len(extracted_str)} bits to {output_path}")

    # --- Verification ---
    embedded_msg_path = os.path.join(args.save_dir, "message_embedded.txt")
    try:
        with open(embedded_msg_path, 'r') as f:
            original_embedded_msg = f.read().strip()
        
        with open(os.path.join(args.save_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        actual_len = metadata['message_len']
        original_embedded_msg = original_embedded_msg[:actual_len]

        if original_embedded_msg == extracted_str:
            logger.log("SUCCESS: Extracted message matches the original embedded message.")
        else:
            logger.log("FAILURE: Extracted message does not match.")
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Could not verify message: {e}")

# ==============================================================================

def main():
    args = create_argparser().parse_args()
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    logger.configure(dir=args.save_dir)
    
    if args.sts_constraint_height < 7 or args.sts_constraint_height > 12:
         raise ValueError(f"Constraint height 'h' must be between 7 and 12.")
    
    if args.sts_payload and len(args.sts_payload) > 7:
        raise ValueError("sts_payload can have at most 7 values, for planes 1-7.")

    if args.mode == 'embed':
        logger.log("Mode: Embedding with full STS algorithm")
        with open(args.message_input, 'r') as f: message = f.read().strip()
        if not all(c in '01' for c in message): raise ValueError("Message contains non-binary characters.")
        
        if args.sts_payload:
            sts_payload_size = sum(args.sts_payload) * 3
        else:
            sts_payload_size = 0 # No payload if not specified
        
        if len(message) < sts_payload_size:
            logger.warn(f"Message length ({len(message)}) is less than required payload ({sts_payload_size}). Padding with zeros.")
            message = message.ljust(sts_payload_size, '0')

        message_to_embed = message[:sts_payload_size]
        
        embedded_msg_path = os.path.join(args.save_dir, "message_embedded.txt")
        with open(embedded_msg_path, 'w') as f: f.write(message_to_embed)
        logger.log(f"Message to be embedded ({len(message_to_embed)} bits) saved.")

        stego_image_np, metadata, cover_image_np = embed_message_sts(args, message_to_embed)
        
        Image.fromarray(cover_image_np, 'RGB').save(os.path.join(args.save_dir, "cover.png"))
        logger.log(f"Cover image saved.")

        Image.fromarray(stego_image_np, 'RGB').save(os.path.join(args.save_dir, "stego.png"))
        logger.log(f"Stego image saved.")
        with open(os.path.join(args.save_dir, "metadata.json"), 'w') as f: json.dump(metadata, f, indent=4)
        logger.log(f"Metadata saved.")

    elif args.mode == 'extract':
        logger.log("Mode: Extracting with full STS algorithm")
        extract_message_sts(args)
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main() 