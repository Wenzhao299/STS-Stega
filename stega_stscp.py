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
# To completely disable logging and file creation from the guided_diffusion library,
# we replace its logger with our dummy logger at the module level.
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
from stscp_sampler import STSSampler
import json
import multiprocessing

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

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

    # Vectorize the calculation for all pixels at once
    val_range = np.arange(256)
    for i in range(n_bits_per_pixel):
        if i < 8: # R channel
            channel_idx = 0
            bit_in_channel = i
        elif i < 16: # G channel
            channel_idx = 1
            bit_in_channel = i - 8
        else: # B channel
            channel_idx = 2
            bit_in_channel = i - 16

        # The mask corresponds to the bit's position within the 8-bit channel value
        # LSB-first logic: plane 0 is LSB of R, plane 7 is MSB of R.
        # bit_in_channel 0..7 maps to bit position 0..7 (LSB to MSB)
        mask = 1 << bit_in_channel
        
        # Probabilities for bit being 0 or 1
        p0_mask = (val_range & mask) == 0
        p1_mask = ~p0_mask

        # Sum probabilities for all pixels at once for the given channel
        # probs_maps[channel_idx, :, :] has shape (num_pixels, 256)
        p0 = np.sum(probs_maps[channel_idx, :, :] * p0_mask, axis=1)
        p1 = np.sum(probs_maps[channel_idx, :, :] * p1_mask, axis=1)
        
        # Store P(s_i=0) and P(s_i=1)
        all_marginal_probs[:, i, 0] = p0
        all_marginal_probs[:, i, 1] = p1

    return all_marginal_probs

def calculate_bit_plane_entropies(all_marginal_probs: np.ndarray) -> np.ndarray:
    """
    Calculates the total Shannon entropy for each of the 24 bit-planes.
    
    Args:
        all_marginal_probs (np.ndarray): A (num_pixels, 24, 2) array of marginal probabilities.

    Returns:
        np.ndarray: A (24,) array containing the total entropy for each bit-plane.
    """
    # Add a small epsilon to prevent log2(0)
    probs = all_marginal_probs + 1e-9
    
    # H(X) = -sum(p(x) * log2(p(x)))
    # Calculate entropy for each bit position
    entropies_per_bit = -np.sum(probs * np.log2(probs), axis=2) # Shape: (num_pixels, 24)
    
    # Sum the entropies of all bits within each plane
    total_plane_entropies = np.sum(entropies_per_bit, axis=0) # Shape: (24,)
    
    return total_plane_entropies

def calculate_conditional_entropies(conditional_probs: np.ndarray) -> np.ndarray:
    """
    Calculates the total Shannon entropy for each bit-plane based on pre-calculated
    conditional probabilities P(s_i | context). This provides a more accurate
    measure of embedding capacity than simple marginal probabilities.
    
    Args:
        conditional_probs (np.ndarray): A (num_pixels, 24, 2) array of conditional probabilities.

    Returns:
        np.ndarray: A (24,) array containing the total entropy for each bit-plane.
    """
    # Add a small epsilon to prevent log2(0)
    probs = conditional_probs + 1e-9
    
    # H(X) = -sum(p(x) * log2(p(x)))
    # Calculate entropy for each bit position for each pixel
    entropies_per_bit = -np.sum(probs * np.log2(probs), axis=2) # Shape: (num_pixels, 24)
    
    # Sum the entropies of all bits within each plane
    total_plane_entropies = np.sum(entropies_per_bit, axis=0) # Shape: (24,)
    
    return total_plane_entropies

def calculate_conditional_probs_for_plane(
    plane_idx: int,
    probs_maps: np.ndarray,
    current_stego_planes: np.ndarray
) -> np.ndarray:
    """
    Calculates the conditional probability P(s_i=b | context) for a given bit-plane.
    The context is the set of already determined lower-order bits in the same channel.
    
    Args:
        plane_idx (int): The index (0-23) of the plane to calculate probabilities for.
        probs_maps (np.ndarray): The original (3, H*W, 256) probability maps from diffusion.
        current_stego_planes (np.ndarray): The (24, H*W) array of bit-planes in their current state.
        
    Returns:
        np.ndarray: A (H*W, 2) array of conditional probabilities for the plane.
    """
    num_pixels = probs_maps.shape[1]
    
    # 1. Determine channel and bit position within the channel
    # R: 0-7, G: 8-15, B: 16-23
    if plane_idx < 8:
        channel_idx = 0
        bit_in_channel = plane_idx
    elif plane_idx < 16:
        channel_idx = 1
        bit_in_channel = plane_idx - 8
    else:
        channel_idx = 2
        bit_in_channel = plane_idx - 16
    
    # 2. Get the relevant channel's probability map
    channel_probs_map = probs_maps[channel_idx]  # Shape: (num_pixels, 256)
    
    # 3. Build the context from already determined lower bits of the same channel
    value_range = np.arange(256, dtype=np.uint8)[None, :] # Shape: (1, 256)
    context_match_mask = np.ones((num_pixels, 256), dtype=bool)

    for b in range(bit_in_channel):
        lower_plane_idx_in_channel = channel_idx * 8 + b
        
        # Get the determined bit values for this lower plane
        lower_plane_bits = current_stego_planes[lower_plane_idx_in_channel, :][:, None] # Shape: (num_pixels, 1)
        
        # Check which of the 256 values match this bit
        plane_b_match = ((value_range >> b) & 1) == lower_plane_bits
        context_match_mask &= plane_b_match

    # 4. Calculate P(context) = sum of probabilities of all values matching the context
    p_context = np.sum(channel_probs_map * context_match_mask, axis=1, keepdims=True)
    
    # Avoid division by zero. If P(context) is 0, it's an impossible state.
    # We set P(context) to 1 to avoid /0 error. The resulting conditional prob will be 0.
    p_context[p_context < 1e-9] = 1.0

    # 5. Calculate P(B_i=1 and context)
    b_is_1_mask = ((value_range >> bit_in_channel) & 1) == 1
    p_b1_and_context = np.sum(channel_probs_map * context_match_mask * b_is_1_mask, axis=1, keepdims=True)
    
    # 6. Calculate conditional probability P(B_i=1 | context)
    p_b1_cond = (p_b1_and_context / p_context).squeeze()
    # Clip to avoid floating point inaccuracies slightly outside [0,1]
    p_b1_cond = np.clip(p_b1_cond, 0, 1)
    p_b0_cond = 1.0 - p_b1_cond

    return np.stack([p_b0_cond, p_b1_cond], axis=1)

def deconstruct_image_to_bit_planes(image_np: np.ndarray) -> np.ndarray:
    """Helper function to deconstruct an RGB image into its 24 bit-planes."""
    h, w, _ = image_np.shape
    num_pixels = h * w
    bit_planes = np.zeros((24, num_pixels), dtype=int)
    for channel_idx in range(3):
        channel_flat = image_np[:, :, channel_idx].flatten()
        for bit_idx in range(8):
            plane_idx = channel_idx * 8 + bit_idx
            bit_planes[plane_idx, :] = (channel_flat >> bit_idx) & 1
    return bit_planes

@torch.no_grad()
def run_diffusion(args):
    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu", weights_only=True)
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    logger.log("loading data...")
    data_loader = load_reference( 
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    logger.log("creating sample...")
    model_kwargs = next(data_loader) 
    model_kwargs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in model_kwargs.items()}
    output = diffusion.p_sample_loop(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        index=1,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        resizers=resizers,
        range_t=args.range_t,
        seed=args.seed,
    )
    logger.log("diffusion complete")
    return output.get('sample'), output.get('mu'), output.get('sigma')

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
        save_dir="output_stscp", 
        seed=1,
        message_input="message.txt",
        # message_output="output_stsc/message_extracted.txt",
        # image_output="output_stsc/stego.png", 
        # image_input="output_stsc/stego.png",  
        mode="embed", 
        sts_payload_rate=3, # Bits per pixel for the entire image
        sts_constraint_height=7, # Default constraint height, must be between 7 and 12
        sts_block_len=65536, # The width of the STC matrix H. Must be a divisor of image_size*image_size.
        save_posterior_probs=False,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    for action in parser._actions:
        if action.dest == 'mode':
            action.choices = ['embed', 'extract', 'generate']
            break
    return parser

def generate_image(args):
    logger.log("Running diffusion to generate a clean image...")
    clean_image_tensor, _, _ = run_diffusion(args)
    if clean_image_tensor is None:
        logger.error("Image generation failed.")
        return
    
    img_tensor = (clean_image_tensor[0].cpu() + 1.0) / 2.0 * 255.0
    img_np = img_tensor.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np, 'RGB')
    save_path = os.path.join(args.save_dir, "cover.png")
    img.save(save_path)
    logger.log(f"Clean cover image saved to {save_path}")

def embed_plane_worker(args_tuple):
    """A worker function for multiprocessing to embed a message chunk into a single bit-plane, block by block."""
    plane_idx, total_payload, constraint_h, num_pixels, block_len, seed, plane_marginal_probs, message_chunk, save_posterior_probs = args_tuple

    # This function runs in a separate process.
    if block_len > num_pixels or num_pixels % block_len != 0:
        # This check is also done in the main thread, but good to have here as a safeguard.
        raise ValueError(f"Invalid block_len={block_len} for num_pixels={num_pixels}.")

    # If payload is zero, we do a wet-only run. The logic is simpler.
    if total_payload <= 0:
        modified_plane = np.argmax(plane_marginal_probs, axis=1)
        posterior_probs = plane_marginal_probs if save_posterior_probs else None
        return plane_idx, modified_plane, posterior_probs, 0

    # Main logic for embedding with a payload
    num_blocks = num_pixels // block_len
    
    # Distribute payload among blocks
    payloads_per_block = [total_payload // num_blocks] * num_blocks
    for i in range(total_payload % num_blocks):
        payloads_per_block[i] += 1
    
    modified_plane_full = np.zeros(num_pixels, dtype=int)
    posterior_probs_full = np.zeros((num_pixels, 2), dtype=float) if save_posterior_probs else None
    
    message_offset = 0
    for i in range(num_blocks):
        start_idx = i * block_len
        end_idx = (i + 1) * block_len
        
        block_probs = plane_marginal_probs[start_idx:end_idx]
        
        payload_size = payloads_per_block[i]
        block_message_chunk = message_chunk[message_offset : message_offset + payload_size]
        message_offset += payload_size

        # Each block gets a unique seed
        worker_seed = seed + plane_idx * num_blocks + i

        if payload_size == 0:
             # This block is wet-only (if payload distribution results in 0 for some blocks)
             modified_plane, posterior_probs = np.argmax(block_probs, axis=1), (block_probs if save_posterior_probs else None)
        else:
            sampler = STSSampler(c=payload_size, h=constraint_h, n=block_len, seed=worker_seed)
            modified_plane, posterior_probs = sampler.sample_bit_plane(block_probs, block_message_chunk, calculate_posterior=save_posterior_probs)

        if modified_plane is None:
            # Propagate failure for the whole plane
            return plane_idx, None, None, total_payload

        modified_plane_full[start_idx:end_idx] = modified_plane
        if save_posterior_probs:
            posterior_probs_full[start_idx:end_idx] = posterior_probs

    return plane_idx, modified_plane_full, posterior_probs_full, total_payload

def generate_probabilities(args):
    """Runs the diffusion model once to get the base probabilities for an image."""
    logger.log("Running diffusion model to get probabilities...")
    _, mu_norm, sigma_norm = run_diffusion(args)
    if mu_norm is None or sigma_norm is None:
        raise RuntimeError("Diffusion model did not return mu and sigma.")

    h, w = args.image_size, args.image_size
    mu_flat_norm = mu_norm.cpu().numpy().flatten()
    sigma_flat_norm = sigma_norm.cpu().numpy().flatten()

    _, probs_maps_list = get_probs_indices_from_diffu(mu_flat_norm, sigma_flat_norm, 3, h, w)
    probs_maps = np.array(probs_maps_list)

    logger.log("Pre-computing marginal probabilities for all bit-planes...")
    all_marginal_probs = get_all_marginal_probs(probs_maps)
    logger.log("Pre-computation of marginal probabilities finished.")
    return probs_maps, all_marginal_probs

def embed_with_probs(args, all_marginal_probs: np.ndarray, message_str: str, use_parallel_planes: bool = True, pbar: Optional[tqdm] = None, verbose: bool = True, plane_processes: Optional[int] = None):
    """
    Embeds a message using pre-computed probabilities. Can run plane embedding in parallel or sequentially.
    """
    h, w = args.image_size, args.image_size
    num_pixels = h * w
    block_len = args.sts_block_len

    # --- Prepare message and stego data structures ---
    metadata = {
        'message_len': total_msg_len, 
        'sts_constraint_height': args.sts_constraint_height, 
        'sts_block_len': block_len,
        'seed': args.seed,
        'payloads_per_plane': payloads_per_plane.tolist(),
    }
    metadata_path = os.path.join(args.save_dir, f"stego_metadata_{args.sts_block_len}_{args.sts_payload_rate:g}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    if num_pixels % block_len != 0:
        raise ValueError(f"Total number of pixels ({num_pixels}) must be divisible by STC block length ({block_len}).")

    # --- Adaptive Payload Distribution based on Entropy ---
    plane_entropies = calculate_bit_plane_entropies(all_marginal_probs)
    total_entropy = np.sum(plane_entropies)
    if total_entropy == 0:
        raise ValueError("Total entropy of the cover is zero. Cannot embed any message.")

    message_bits = np.array([int(b) for b in message_str], dtype=int)
    total_msg_len = len(message_bits)
    logger.log(f"Total message length: {total_msg_len} bits.")
    
    # --- New Proportional Payload Distribution Logic ---
    plane_capacities = np.floor(plane_entropies * 0.95).astype(int)
    total_capacity = np.sum(plane_capacities)

    if total_capacity == 0:
        raise ValueError("Total embedding capacity of the image is zero.")
    if total_msg_len > total_capacity:
        raise ValueError(f"Message is too long ({total_msg_len} bits) for image capacity ({total_capacity} bits).")

    # Distribute the total payload proportionally to each plane's capacity
    payloads_per_plane = np.floor(total_msg_len * (plane_capacities / total_capacity)).astype(int)
    
    # The proportional distribution might leave a few bits due to flooring.
    # Distribute the remainder greedily to the planes with the highest capacity.
    remainder = total_msg_len - np.sum(payloads_per_plane)
    if remainder > 0:
        # Get indices of planes sorted by capacity, descending
        sorted_plane_indices = np.argsort(-plane_capacities)
        for i in range(remainder):
            plane_to_add = sorted_plane_indices[i % len(sorted_plane_indices)]
            payloads_per_plane[plane_to_add] += 1
    
    # --- End of New Logic ---

    logger.log(f"Payloads per plane determined for rate {args.sts_payload_rate:g}: {payloads_per_plane.tolist()}")

    # --- Prepare for parallel workers ---
    message_offset = 0
    plane_to_chunk_map = {}
    for plane_idx in range(24):
        payload = payloads_per_plane[plane_idx]
        if payload > 0:
            plane_to_chunk_map[plane_idx] = message_bits[message_offset : message_offset + payload]
            message_offset += payload

    worker_args = []
    for plane_idx in range(24):
        payload_for_plane = payloads_per_plane[plane_idx]
        message_chunk = plane_to_chunk_map.get(plane_idx, np.array([], dtype=int))
        plane_marginal_probs = all_marginal_probs[:, plane_idx, :]
        worker_args.append((
            plane_idx, payload_for_plane, args.sts_constraint_height, 
            num_pixels, block_len, args.seed, 
            plane_marginal_probs, message_chunk,
            args.save_posterior_probs
        ))
    
    # --- Run embedding for each bit-plane ---
    modified_planes = np.zeros((24, num_pixels), dtype=int)
    posterior_probs_planes = np.zeros_like(all_marginal_probs) if args.save_posterior_probs else None
    total_embedded_payload = 0
    failed_planes = []

    iterator = None
    desc = ""

    if use_parallel_planes:
        logger.log("Starting parallel embedding process for all bit-planes...")
        num_procs = plane_processes if plane_processes is not None else (os.cpu_count() or 1)
        # We need to manage the pool manually to use it in the loop
        pool = multiprocessing.Pool(processes=min(24, num_procs))
        iterator = pool.imap_unordered(embed_plane_worker, worker_args)
        desc = "Embedding into planes (Parallel)"
    else:
        logger.log("Starting sequential embedding process for all bit-planes...")
        iterator = (embed_plane_worker(arg) for arg in worker_args)
        desc = "Embedding into planes (Sequential)"

    # --- Unified Loop for Processing Results ---
    iterator_to_run = iterator
    # If an external pbar is NOT provided, but verbosity is on, create an internal tqdm instance
    if not pbar and verbose:
        iterator_to_run = tqdm(iterator, total=len(worker_args), desc=desc)
    
    # If an external pbar IS provided, set its total. We will update it manually.
    if pbar:
        pbar.total = len(worker_args)

    for result in iterator_to_run:
        plane_idx, modified_plane, posterior_probs, embedded_payload = result
        if modified_plane is None:
            failed_planes.append((plane_idx, embedded_payload))
        else:
            modified_planes[plane_idx, :] = modified_plane
            if args.save_posterior_probs and posterior_probs is not None:
                posterior_probs_planes[:, plane_idx, :] = posterior_probs
            total_embedded_payload += embedded_payload
        
        # If an external pbar was passed, we must update it manually.
        # If an internal one was created, tqdm handles the updates automatically.
        if pbar:
            pbar.update(1)

    # Clean up the pool if it was created
    if use_parallel_planes and pool:
        pool.close()
        pool.join()
    
    if failed_planes:
        error_msg = f"Embedding failed for {len(failed_planes)} planes."
        for p_idx, p_load in failed_planes: error_msg += f"\n  - Plane {p_idx} failed to embed {p_load} bits."
        raise RuntimeError(error_msg)

    logger.log(f"Successfully embedded {total_embedded_payload} bits in total.")

    # --- Reconstruct the stego image from the modified bit-planes ---
    logger.log("Reconstructing stego image from bit-planes...")
    stego_image_np = np.zeros((h, w, 3), dtype=np.uint8)
    
    for channel_idx in range(3):
        val = np.zeros(num_pixels, dtype=np.int32)
        for bit_idx in range(8):
            plane_idx = channel_idx * 8 + bit_idx
            val += modified_planes[plane_idx].astype(np.int32) << bit_idx
        stego_image_np[:, :, channel_idx] = val.reshape((h, w))
        
    # --- Save Stego Image and Probabilities ---
    if args.save_posterior_probs:
        stego_probs_save_path = os.path.join(args.save_dir, f"stego_marginal_probs_{args.sts_block_len}_{args.sts_payload_rate:g}.npy")
        np.save(stego_probs_save_path, posterior_probs_planes)
    img = Image.fromarray(stego_image_np, 'RGB')
    stego_path = os.path.join(args.save_dir, f"stego_{args.sts_block_len}_{args.sts_payload_rate:g}.png")
    img.save(stego_path)
    logger.log(f"Stego image saved to {stego_path}")
    logger.log(f"Embedding finished. {total_embedded_payload} bits embedded.")

    return stego_image_np

def reconstruct_image_from_bit_planes(modified_planes: np.ndarray, h: int, w: int) -> np.ndarray:
    """Helper function to reconstruct an RGB image from its 24 bit-planes."""
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
    """
    Generates a cover image by sampling a value for each pixel channel from its
    256-value probability distribution.
    """
    num_pixels = h * w
    
    # Vectorized sampling using inverse transform sampling
    # 1. Calculate cumulative distribution function (CDF) for each distribution
    cdfs = np.cumsum(probs_maps, axis=2)

    # 2. Generate uniform random numbers for each pixel and channel
    uniform_samples = rng.random((3, num_pixels, 1))
    
    # 3. Find the first index where the uniform sample is less than the CDF value.
    #    This is equivalent to finding the sampled value.
    sampled_indices = np.argmax(uniform_samples < cdfs, axis=2).astype(np.uint8)

    # 4. Reshape to image dimensions
    cover_image_np = np.zeros((h, w, 3), dtype=np.uint8)
    cover_image_np[:, :, 0] = sampled_indices[0, :].reshape((h, w))
    cover_image_np[:, :, 1] = sampled_indices[1, :].reshape((h, w))
    cover_image_np[:, :, 2] = sampled_indices[2, :].reshape((h, w))

    return cover_image_np

def calculate_cover_channel_conditional_probs_worker(args_tuple):
    """
    A worker to calculate the conditional probabilities for a "clean" cover image channel.
    This mimics the STScp probability calculation process but without any embedding.
    The resulting probabilities P(s_i|context) are the proper baseline for KL divergence.
    """
    (
        channel_idx,
        probs_maps,
        template_cover_bits_np,
        image_size
    ) = args_tuple

    num_pixels = image_size * image_size
    
    # This structure is local to the worker and will store the calculated conditional probabilities.
    cover_cond_probs_channel_np = np.zeros((num_pixels, 8, 2))

    # Process from LSB (bit 0) up to MSB (bit 7) for the assigned channel.
    # The context is always taken from the unmodified template cover bit planes.
    # pbar_channel = tqdm(range(8), desc=f"Cover Prob Chan {channel_idx}", position=channel_idx, leave=True)
    pbar_channel = range(8)
    for bit_in_channel in pbar_channel:
        plane_idx = channel_idx * 8 + bit_in_channel
        
        # Calculate conditional probabilities for the current plane based on the
        # context from the *original template cover's* lower bit planes.
        cond_probs = calculate_conditional_probs_for_plane(
            plane_idx, probs_maps, template_cover_bits_np
        )
        
        # Store the result for this plane.
        cover_cond_probs_channel_np[:, bit_in_channel, :] = cond_probs
    
    #  pbar_channel.close()
    return channel_idx, cover_cond_probs_channel_np

def embed_channel_stscp_worker(args_tuple):
    """
    A worker function for multiprocessing to embed message bits into all planes of a single channel.
    It respects the LSB->MSB dependency within the channel.
    Returns the modified bit planes and their posterior probabilities for that channel.
    """
    (
        channel_idx,
        args,  # The main script args object
        probs_maps,  # The full (3, H*W, 256) array, shared by COW
        template_cover_bits_np,  # The full (24, H*W) template bits, shared by COW
        payloads_per_plane,  # The full (24,) array
        plane_to_chunk_map,  # The full map from plane_idx to its message chunk
    ) = args_tuple

    h, w = args.image_size, args.image_size
    num_pixels = h * w
    block_len = args.sts_block_len

    # The worker operates on a local copy of the bit planes. This is crucial for maintaining
    # the correct context as we modify planes from LSB to MSB.
    stego_bits_for_worker = np.copy(template_cover_bits_np)
    
    # These dictionaries will ONLY store the data for the planes that are actually modified.
    modified_bits_dict = {}
    posterior_probs_dict = {}

    # Process from LSB (bit 0) up to MSB (bit 7) for the assigned channel.
    bit_indices_in_channel = range(8)
    # pbar_channel = tqdm(bit_indices_in_channel, desc=f"Channel {channel_idx}", position=channel_idx, leave=True)
    pbar_channel = bit_indices_in_channel

    for bit_in_channel in pbar_channel:
        plane_idx = channel_idx * 8 + bit_in_channel
        payload_for_plane = payloads_per_plane[plane_idx]

        # 1. Calculate conditional probabilities based on the CURRENT state of lower bits.
        # This is now done for ALL planes, regardless of payload, to ensure correct context.
        cond_probs = calculate_conditional_probs_for_plane(
            plane_idx, probs_maps, stego_bits_for_worker
        )

        modified_plane = None
        posterior_probs_plane = None

        if payload_for_plane == 0:
            # WET RUN: No payload. Sample bits based on conditional probability to create
            # a new random instance, independent of the cover image's sample.
            worker_rng_seed = args.seed + 1000 * channel_idx + 100 * bit_in_channel
            rng = np.random.default_rng(worker_rng_seed)
            uniform_samples = rng.random(num_pixels)
            # Sample bit=1 if random sample > P(bit=0), else bit=0
            modified_plane = (uniform_samples > cond_probs[:, 0]).astype(int)

            # For a wet-run, the posterior probability is simply the conditional probability.
            if args.save_posterior_probs:
                posterior_probs_plane = cond_probs
        
        else: # DRY RUN: This logic only runs for planes with a payload.
            message_chunk = plane_to_chunk_map.get(plane_idx, np.array([], dtype=int))
            num_blocks = num_pixels // block_len
            payloads_per_block = [payload_for_plane // num_blocks] * num_blocks
            for i in range(payload_for_plane % num_blocks):
                payloads_per_block[i] += 1
            
            # Temporary arrays for this plane's results
            modified_plane_temp = np.zeros(num_pixels, dtype=int)
            posterior_probs_plane_temp = np.zeros((num_pixels, 2)) if args.save_posterior_probs else None
            msg_offset_in_plane = 0
            
            embedding_successful = True
            for i in range(num_blocks):
                start_idx = i * block_len
                end_idx = (i + 1) * block_len
                block_probs = cond_probs[start_idx:end_idx]
                
                payload_size = payloads_per_block[i]
                block_message_chunk = message_chunk[msg_offset_in_plane : msg_offset_in_plane + payload_size]
                msg_offset_in_plane += payload_size
                
                worker_seed = args.seed + plane_idx * num_blocks + i

                sampler = STSSampler(c=payload_size, h=args.sts_constraint_height, n=block_len, seed=worker_seed)
                modified_block, posterior_probs_block = sampler.sample_bit_plane(
                    block_probs, block_message_chunk, calculate_posterior=args.save_posterior_probs, verbose=False
                )

                if modified_block is None:
                    embedding_successful = False
                    break 

                modified_plane_temp[start_idx:end_idx] = modified_block
                if args.save_posterior_probs and posterior_probs_block is not None:
                    posterior_probs_plane_temp[start_idx:end_idx, :] = posterior_probs_block

            if not embedding_successful:
                # pbar_channel.close()
                return channel_idx, None, None, plane_idx # Propagate failure
            
            modified_plane = modified_plane_temp
            posterior_probs_plane = posterior_probs_plane_temp

        # 3. Store the results for this plane in the dictionaries.
        modified_bits_dict[plane_idx] = modified_plane
        if args.save_posterior_probs and posterior_probs_plane is not None:
            posterior_probs_dict[plane_idx] = posterior_probs_plane

        # 4. CRUCIALLY, update the worker's local state with the newly modified plane.
        #    This provides the correct context for the next iteration (a lower bit-plane).
        stego_bits_for_worker[plane_idx, :] = modified_plane
    
    # pbar_channel.close()
    
    # Return the dictionaries of modified data. They will be empty if no payload was in this channel.
    return channel_idx, modified_bits_dict, posterior_probs_dict, None

def embed_message_stscp(args, message_str: str):
    """
    Embeds a message using the STScp algorithm:
    1. Generates a template cover image via direct sampling.
    2. Uses this template for non-embedded bits.
    3. For embedding planes, calculates conditional probabilities based on determined
       lower bits and then uses STS sampler.
    """
    logger.log("STScp embedding with conditional probabilities started.")
    h, w = args.image_size, args.image_size
    num_pixels = h * w
    block_len = args.sts_block_len
    
    # --- 1. Generate probabilities and template cover ---
    probs_maps, all_marginal_probs = generate_probabilities(args)
    rng = np.random.default_rng(args.seed)
    template_cover_np = generate_cover_from_probs(probs_maps, h, w, rng)
    logger.log("Generated base probabilities and template cover image.")

    # --- Save original conditional marginals for analysis ---
    original_probs_save_path = os.path.join(args.save_dir, "original_marginal_probs.npy")

    if not os.path.exists(original_probs_save_path):
        logger.log("Calculating conditional probabilities for the original cover image...")
        template_cover_bits_np_for_prob_calc = deconstruct_image_to_bit_planes(template_cover_np)
        
        cover_prob_worker_args = []
        for i in range(3):
            cover_prob_worker_args.append((i, probs_maps, template_cover_bits_np_for_prob_calc, args.image_size))

        num_procs_cover = min(3, os.cpu_count() or 1)
        cover_conditional_probs_np = np.zeros((num_pixels, 24, 2))

        with multiprocessing.Pool(processes=num_procs_cover) as pool:
            results = list(pool.imap_unordered(calculate_cover_channel_conditional_probs_worker, cover_prob_worker_args))
        
        for channel_idx, channel_cond_probs in results:
            # channel_cond_probs has shape (num_pixels, 8, 2)
            for bit_in_channel in range(8):
                plane_idx = channel_idx * 8 + bit_in_channel
                cover_conditional_probs_np[:, plane_idx, :] = channel_cond_probs[:, bit_in_channel, :]
        
        np.save(original_probs_save_path, cover_conditional_probs_np)
        logger.log(f"Original conditional marginal probabilities saved to {original_probs_save_path}")
    else:
        logger.log(f"Original conditional marginal probabilities already exist at {original_probs_save_path}, skipping calculation.")

    # Load the definitive original probabilities for use as a baseline
    original_conditional_probs_np = np.load(original_probs_save_path)

    # --- 2. Deconstruct template into initial bit planes for steganography ---
    template_cover_bits_np = deconstruct_image_to_bit_planes(template_cover_np)

    # --- 3. Determine payload distribution using the simple marginals ---
    plane_entropies = calculate_bit_plane_entropies(all_marginal_probs)

    message_bits = np.array([int(b) for b in message_str], dtype=int)
    total_msg_len = len(message_bits)

    filling_order = []
    for bit_idx in range(8):
        for channel_idx in range(3):
            # The plane index is now calculated based on LSB-first processing within a layer
            plane_idx = channel_idx * 8 + bit_idx
            filling_order.append(plane_idx)
    
    # Use the hierarchical distribution method to match the batch processing logic
    payloads_per_plane = distribute_payload_hierarchically(total_msg_len, plane_entropies)

    logger.log(f"Payloads per plane determined for rate {args.sts_payload_rate:g}: {payloads_per_plane.tolist()}")

    # --- Prepare for parallel workers (same as before) ---
    message_offset = 0
    plane_to_chunk_map = {}
    for plane_idx in filling_order:
        payload = payloads_per_plane[plane_idx]
        if payload > 0:
            plane_to_chunk_map[plane_idx] = message_bits[message_offset : message_offset + payload]
            message_offset += payload

    worker_args_list = []
    for channel_idx in range(3):
        worker_args_list.append((
            channel_idx,
            args,
            probs_maps,
            template_cover_bits_np, # Pass the initial template bits
            payloads_per_plane,
            plane_to_chunk_map
        ))

    # --- 5. Iteratively embed, with each channel running in parallel ---
    logger.log("Starting parallel embedding process for each channel...")
    num_procs = min(3, os.cpu_count() or 1)
    
    # Initialize final stego data structures.
    # The bits start as a direct copy of the template cover.
    # The probabilities start as a direct copy of the original conditional probabilities.
    final_stego_bits_np = np.copy(template_cover_bits_np)
    final_stego_marginal_probs_np = np.copy(original_conditional_probs_np)
    
    with multiprocessing.Pool(processes=num_procs) as pool:
        results = list(pool.imap_unordered(embed_channel_stscp_worker, worker_args_list))

    # --- 6. Process results from workers ---
    failed_channels = []
    for result in results:
        channel_idx, modified_bits_dict, posterior_probs_dict, failed_plane_idx = result
        if modified_bits_dict is None: # Indicates a failure inside the worker
            failed_channels.append((channel_idx, failed_plane_idx))
        else:
            # Update the final arrays ONLY with the data from the modified planes.
            # Planes not in these dictionaries remain untouched.
            if modified_bits_dict:
                for plane_idx, bits in modified_bits_dict.items():
                    final_stego_bits_np[plane_idx, :] = bits
            
            if posterior_probs_dict:
                for plane_idx, probs in posterior_probs_dict.items():
                    final_stego_marginal_probs_np[:, plane_idx, :] = probs

    if failed_channels:
        error_msg = f"Embedding failed for {len(failed_channels)} channels."
        for c_idx, p_idx in failed_channels:
            error_msg += f"\n  - Channel {c_idx} failed at plane {p_idx}."
        raise RuntimeError(error_msg)

    logger.log("All channels processed successfully.")

    # --- 7. Reconstruct and Save ---
    logger.log("Reconstructing final stego image...")
    stego_image_np = reconstruct_image_from_bit_planes(final_stego_bits_np, h, w)
    
    img = Image.fromarray(stego_image_np, 'RGB')
    stego_path = os.path.join(args.save_dir, f"stego_{args.sts_block_len}_{args.sts_payload_rate:g}.png")
    img.save(stego_path)
    logger.log(f"STScp stego image saved to {stego_path}")
    
    # Save posterior probabilities if calculated
    if args.save_posterior_probs:
        stego_probs_save_path = os.path.join(args.save_dir, f"stego_marginal_probs_{args.sts_block_len}_{args.sts_payload_rate:g}.npy")
        np.save(stego_probs_save_path, final_stego_marginal_probs_np)
        logger.log(f"Stego posterior marginal probabilities saved to {stego_probs_save_path}")

    # Save metadata
    metadata = {
        'message_len': int(total_msg_len), 
        'sts_constraint_height': int(args.sts_constraint_height), 
        'sts_block_len': int(block_len),
        'seed': int(args.seed),
        'payloads_per_plane': [int(p) for p in payloads_per_plane],
        'algorithm': 'stscp' # Add identifier for the algorithm
    }
    metadata_path = os.path.join(args.save_dir, f"stego_metadata_{args.sts_block_len}_{args.sts_payload_rate:g}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.log(f"STScp metadata saved to {metadata_path}")
    logger.log(f"STScp embedding finished.")

    return stego_image_np

def distribute_payload_hierarchically(total_msg_len: int, plane_entropies: np.ndarray) -> np.ndarray:
    """
    Distributes payload hierarchically, filling LSB layers first, and distributing
    proportionally within each layer.
    """
    plane_capacities = np.floor(plane_entropies * 0.95).astype(int)
    total_image_capacity = np.sum(plane_capacities)
    if total_msg_len > total_image_capacity:
        raise ValueError(f"Message ({total_msg_len}) exceeds total image capacity ({total_image_capacity}).")

    payloads_per_plane = np.zeros(24, dtype=int)
    message_left = total_msg_len

    # Iterate layer by layer, from LSB (0) to MSB (7)
    for bit_idx in range(8):
        if message_left <= 0:
            break

        # Define the three planes in the current layer (R, G, B)
        plane_indices = [bit_idx, 8 + bit_idx, 16 + bit_idx]
        
        # Get capacities for just this layer
        capacities_in_layer = plane_capacities[plane_indices]
        total_layer_capacity = np.sum(capacities_in_layer)

        if total_layer_capacity == 0:
            continue

        # Determine how much payload this layer will handle
        payload_for_this_layer = min(message_left, total_layer_capacity)
        
        if payload_for_this_layer > 0:
            # Distribute this layer's payload proportionally among its 3 planes
            proportions = capacities_in_layer / (total_layer_capacity + 1e-9)
            distributed_payloads = np.floor(payload_for_this_layer * proportions).astype(int)
            
            # Handle remainder from flooring by giving it to the most capable planes in the layer
            remainder = payload_for_this_layer - np.sum(distributed_payloads)
            if remainder > 0:
                sorted_indices_in_layer = np.argsort(-capacities_in_layer)
                for i in range(remainder):
                    plane_to_add_in_layer = sorted_indices_in_layer[i % len(sorted_indices_in_layer)]
                    distributed_payloads[plane_to_add_in_layer] += 1
            
            payloads_per_plane[plane_indices] = distributed_payloads
        
        message_left -= payload_for_this_layer
        
    return payloads_per_plane

def perform_embedding(args, message_str, probs_maps, all_marginal_probs, template_cover_np, original_conditional_probs_np):
    """
    Performs the core embedding logic using pre-computed probability data and a
    pre-generated template cover. Meant to be called in a loop for batch processing.
    """
    h, w = args.image_size, args.image_size
    
    # The original conditional probabilities are now passed directly as an argument.
    # This avoids reading the file from disk in every single call.
    # original_probs_save_path = os.path.join(args.save_dir, "original_marginal_probs.npy")
    # original_conditional_probs_np = np.load(original_probs_save_path)

    # Deconstruct the template cover for use in embedding
    template_cover_bits_np = deconstruct_image_to_bit_planes(template_cover_np)

    # --- Payload Distribution ---
    plane_entropies = calculate_bit_plane_entropies(all_marginal_probs)
    payloads_per_plane = distribute_payload_hierarchically(len(message_str), plane_entropies)

    logger.log(f"Payloads per plane determined for rate {args.sts_payload_rate:g}: {payloads_per_plane.tolist()}")

    # --- Prepare for parallel workers ---
    message_bits = np.array([int(b) for b in message_str], dtype=int)
    message_offset = 0
    plane_to_chunk_map = {}
    # Message is packed LSB-first
    filling_order = []
    for bit_idx in range(8):
        for channel_idx in range(3):
            # The plane index is now calculated based on LSB-first processing within a layer
            plane_idx = channel_idx * 8 + bit_idx
            filling_order.append(plane_idx)

    for plane_idx in filling_order:
        payload = payloads_per_plane[plane_idx]
        if payload > 0:
            plane_to_chunk_map[plane_idx] = message_bits[message_offset : message_offset + payload]
            message_offset += payload

    worker_args_list = []
    for channel_idx in range(3):
        worker_args_list.append((
            channel_idx, args, probs_maps, template_cover_bits_np,
            payloads_per_plane, plane_to_chunk_map
        ))

    # --- Run parallel embedding ---
    num_procs = min(3, os.cpu_count() or 1)
    final_stego_bits_np = np.copy(template_cover_bits_np)
    final_stego_marginal_probs_np = np.copy(original_conditional_probs_np)
    
    with multiprocessing.Pool(processes=num_procs) as pool:
        results = list(pool.imap_unordered(embed_channel_stscp_worker, worker_args_list))

    # --- Process results ---
    failed_channels = []
    for result in results:
        channel_idx, modified_bits_dict, posterior_probs_dict, failed_plane_idx = result
        if modified_bits_dict is None:
            failed_channels.append((channel_idx, failed_plane_idx))
        else:
            if modified_bits_dict:
                for plane_idx, bits in modified_bits_dict.items():
                    final_stego_bits_np[plane_idx, :] = bits
            if posterior_probs_dict:
                for plane_idx, probs in posterior_probs_dict.items():
                    final_stego_marginal_probs_np[:, plane_idx, :] = probs
    
    if failed_channels:
        raise RuntimeError(f"Embedding failed for channels: {failed_channels}")
    
    logger.log("All channels processed successfully for this run.")

    # --- Reconstruct and Save ---
    stego_image_np = reconstruct_image_from_bit_planes(final_stego_bits_np, h, w)
    
    img = Image.fromarray(stego_image_np, 'RGB')
    stego_path = os.path.join(args.save_dir, f"stego_{args.sts_block_len}_{args.sts_payload_rate:g}.png")
    img.save(stego_path)
    logger.log(f"Stego image saved to {stego_path}")
    
    if args.save_posterior_probs:
        stego_probs_save_path = os.path.join(args.save_dir, f"stego_marginal_probs_{args.sts_block_len}_{args.sts_payload_rate:g}.npy")
        np.save(stego_probs_save_path, final_stego_marginal_probs_np)

    metadata = {
        'message_len': int(len(message_str)),
        'sts_constraint_height': int(args.sts_constraint_height), 
        'sts_block_len': int(args.sts_block_len),
        'seed': int(args.seed),
        'payloads_per_plane': [int(p) for p in payloads_per_plane],
        'algorithm': 'stscp'
    }
    metadata_path = os.path.join(args.save_dir, f"stego_metadata_{args.sts_block_len}_{args.sts_payload_rate:g}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return stego_image_np

def prepare_data_for_pair(args):
    """
    Prepares all necessary data for generating a single cover/stego pair.
    This includes running the diffusion model, generating a template cover,
    and calculating the original conditional probabilities for that cover.
    Meant to be called by a producer in a parallel pipeline.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - probs_maps: The (3, H*W, 256) raw probability maps.
            - all_marginal_probs: The (H*W, 24, 2) simple marginal probabilities.
            - template_cover_np: The (H, W, 3) generated cover image.
            - cover_conditional_probs_np: The (H*W, 24, 2) conditional probabilities of the cover.
    """
    h, w = args.image_size, args.image_size
    num_pixels = h * w
    
    # 1. Run diffusion model to get base probabilities
    probs_maps, all_marginal_probs = generate_probabilities(args)
    
    # 2. Generate the single, definitive template cover image for this pair
    rng = np.random.default_rng(args.seed)
    template_cover_np = generate_cover_from_probs(probs_maps, h, w, rng)

    # 3. Calculate the conditional probabilities of the original cover
    template_cover_bits_np = deconstruct_image_to_bit_planes(template_cover_np)
    
    cover_prob_worker_args = []
    for i in range(3):
        cover_prob_worker_args.append((i, probs_maps, template_cover_bits_np, args.image_size))

    # Use a small pool for this self-contained calculation
    num_procs_cover = min(3, (os.cpu_count() or 1))
    cover_conditional_probs_np = np.zeros((num_pixels, 24, 2))

    # This function might be called from a process that is already a child of a Pool.
    # To avoid errors with daemonic processes, we can use a simple loop or a ThreadPool
    # if full sub-processing causes issues. For CPU-bound tasks, multiprocessing is better.
    # A robust implementation would check if it can create a sub-pool. For now, we assume it can.
    try:
        with multiprocessing.Pool(processes=num_procs_cover) as pool:
            results = list(pool.imap_unordered(calculate_cover_channel_conditional_probs_worker, cover_prob_worker_args))
    except Exception:
        # Fallback to sequential execution if pooling fails (e.g., in a daemon)
        results = [calculate_cover_channel_conditional_probs_worker(arg) for arg in cover_prob_worker_args]

    for channel_idx, channel_cond_probs in results:
        for bit_in_channel in range(8):
            plane_idx = channel_idx * 8 + bit_in_channel
            cover_conditional_probs_np[:, plane_idx, :] = channel_cond_probs[:, bit_in_channel, :]
    
    return probs_maps, all_marginal_probs, template_cover_np, cover_conditional_probs_np

def setup_and_generate_probabilities(args):
    """
    Handles the one-time setup: running diffusion, generating the template cover,
    and calculating the definitive, conditional original probabilities.
    This is meant to be called ONCE per batch experiment.
    """
    logger.log("--- Running Initial Setup and Probability Generation ---")
    h, w = args.image_size, args.image_size
    num_pixels = h * w
    
    # 1. Run diffusion model to get base probabilities
    probs_maps, all_marginal_probs = generate_probabilities(args)
    
    # 2. Generate the single, definitive template cover image for this batch
    rng = np.random.default_rng(args.seed)
    template_cover_np = generate_cover_from_probs(probs_maps, h, w, rng)
    logger.log("Generated base probabilities and the definitive template cover image.")

    # 3. Calculate and save the conditional probabilities of the original cover
    original_probs_save_path = os.path.join(args.save_dir, "original_marginal_probs.npy")

    logger.log("Calculating conditional probabilities for the original cover image...")
    template_cover_bits_np = deconstruct_image_to_bit_planes(template_cover_np)
    
    cover_prob_worker_args = []
    for i in range(3):
        cover_prob_worker_args.append((i, probs_maps, template_cover_bits_np, args.image_size))

    num_procs_cover = min(3, os.cpu_count() or 1)
    cover_conditional_probs_np = np.zeros((num_pixels, 24, 2))

    with multiprocessing.Pool(processes=num_procs_cover) as pool:
        results = list(pool.imap_unordered(calculate_cover_channel_conditional_probs_worker, cover_prob_worker_args))
    
    for channel_idx, channel_cond_probs in results:
        for bit_in_channel in range(8):
            plane_idx = channel_idx * 8 + bit_in_channel
            cover_conditional_probs_np[:, plane_idx, :] = channel_cond_probs[:, bit_in_channel, :]
    
    np.save(original_probs_save_path, cover_conditional_probs_np)
    logger.log(f"Definitive original conditional marginal probabilities saved to {original_probs_save_path}")

    return probs_maps, all_marginal_probs, template_cover_np

def embed_using_precalculated_data(args, message_str: str, probs_maps: np.ndarray, probability_data_for_entropy: np.ndarray, template_cover_np: np.ndarray, original_conditional_probs_np: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Performs the STScp embedding logic using a full set of pre-computed data.
    This is the core embedding logic separated from data generation. It does NOT save files.
    
    Returns:
        A tuple containing:
         - np.ndarray: The final (H, W, 3) stego image as a numpy array.
         - dict: The metadata dictionary for the embedding.
    """
    h, w = args.image_size, args.image_size
    
    # Deconstruct the template cover for use in embedding
    template_cover_bits_np = deconstruct_image_to_bit_planes(template_cover_np)

    # --- Payload Distribution ---
    # Use the new, more accurate conditional entropy calculation
    plane_entropies = calculate_conditional_entropies(probability_data_for_entropy)
    payloads_per_plane = distribute_payload_hierarchically(len(message_str), plane_entropies)

    # --- Prepare for parallel workers ---
    message_bits = np.array([int(b) for b in message_str], dtype=int)
    message_offset = 0
    plane_to_chunk_map = {}
    filling_order = []
    for bit_idx in range(8):
        for channel_idx in range(3):
            # The plane index is now calculated based on LSB-first processing within a layer
            plane_idx = channel_idx * 8 + bit_idx
            filling_order.append(plane_idx)

    for plane_idx in filling_order:
        payload = payloads_per_plane[plane_idx]
        if payload > 0:
            plane_to_chunk_map[plane_idx] = message_bits[message_offset : message_offset + payload]
            message_offset += payload

    worker_args_list = []
    for channel_idx in range(3):
        worker_args_list.append((
            channel_idx, args, probs_maps, template_cover_bits_np,
            payloads_per_plane, plane_to_chunk_map
        ))

    # --- Run parallel embedding ---
    num_procs = min(3, os.cpu_count() or 1)
    final_stego_bits_np = np.copy(template_cover_bits_np)
    final_stego_marginal_probs_np = np.copy(original_conditional_probs_np)
    
    with multiprocessing.Pool(processes=num_procs) as pool:
        results = list(pool.imap_unordered(embed_channel_stscp_worker, worker_args_list))

    # --- Process results ---
    failed_channels = []
    for result in results:
        channel_idx, modified_bits_dict, posterior_probs_dict, failed_plane_idx = result
        if modified_bits_dict is None:
            failed_channels.append((channel_idx, failed_plane_idx))
        else:
            if modified_bits_dict:
                for plane_idx, bits in modified_bits_dict.items():
                    final_stego_bits_np[plane_idx, :] = bits
            if posterior_probs_dict:
                for plane_idx, probs in posterior_probs_dict.items():
                    final_stego_marginal_probs_np[:, plane_idx, :] = probs
    
    if failed_channels:
        raise RuntimeError(f"Embedding failed for channels: {failed_channels}")
    
    # --- Reconstruct stego image ---
    stego_image_np = reconstruct_image_from_bit_planes(final_stego_bits_np, h, w)
    
    # --- Create metadata for return ---
    metadata = {
        'message_len': int(len(message_str)),
        'sts_constraint_height': int(args.sts_constraint_height), 
        'sts_block_len': int(args.sts_block_len),
        'seed': int(args.seed),
        'payloads_per_plane': [int(p) for p in payloads_per_plane],
        'algorithm': 'stscp'
    }
        
    return stego_image_np, metadata

def extract_message_stscp(args):
    """Extracts a message embedded with the STScp algorithm."""
    logger.log("STScp extraction started.")
    # Since the underlying STC mechanism is the same, extraction is identical to the original,
    # just with different filenames for image and metadata.
    stego_path = os.path.join(args.save_dir, f"stego_{args.sts_block_len}_{args.sts_payload_rate:g}.png")
    metadata_path = os.path.join(args.save_dir, f"stego_metadata_{args.sts_block_len}_{args.sts_payload_rate:g}.json")
    output_path = os.path.join(args.save_dir, f"message_extracted_{args.sts_block_len}_{args.sts_payload_rate:g}.txt")
    
    try:
        stego_image_pil = Image.open(stego_path).convert('RGB')
    except FileNotFoundError:
        logger.error(f"STScp stego image not found at {stego_path}")
        return

    stego_image_np = np.array(stego_image_pil)
    h, w, _ = stego_image_np.shape
    num_pixels = h * w
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        message_len = metadata['message_len']
        constraint_height = metadata['sts_constraint_height']
        seed = metadata['seed']
        payloads_per_plane = metadata['payloads_per_plane']
        block_len = metadata.get('sts_block_len', num_pixels)
    except FileNotFoundError:
        logger.error(f"Metadata file '{metadata_path}' not found. Cannot extract message.")
        return
        
    if num_pixels % block_len != 0:
        raise ValueError(f"Total number of pixels ({num_pixels}) is not divisible by the STC block length ({block_len}).")

    num_blocks = num_pixels // block_len

    # --- Deconstruct image into bit-planes ---
    logger.log("Deconstructing stego image into bit-planes...")
    bit_planes = deconstruct_image_to_bit_planes(stego_image_np)

    # --- Extract message from relevant bit-planes ---
    filling_order = []
    for bit_idx in range(8):
        for channel_idx in range(3):
            # The plane index is now calculated based on LSB-first processing within a layer
            plane_idx = channel_idx * 8 + bit_idx
            filling_order.append(plane_idx)
            
    extracted_message_bits = []
    pbar = tqdm(filling_order, desc="Extracting from planes (STScp)")

    for plane_idx in pbar:
        payload_for_plane = payloads_per_plane[plane_idx]
        if payload_for_plane == 0:
            continue
        
        pbar.set_postfix({"plane": plane_idx, "payload": f"{payload_for_plane} bits"})
        stego_plane_bits = bit_planes[plane_idx]

        payloads_per_block = [payload_for_plane // num_blocks] * num_blocks
        for i in range(payload_for_plane % num_blocks):
            payloads_per_block[i] += 1
        
        for i in range(num_blocks):
            payload_size = payloads_per_block[i]
            if payload_size == 0:
                continue

            start_idx = i * block_len
            end_idx = (i + 1) * block_len
            stego_block_bits = stego_plane_bits[start_idx:end_idx]

            worker_seed = seed + plane_idx * num_blocks + i
            sampler = STSSampler(c=payload_size, h=constraint_height, n=block_len, seed=worker_seed)
            H_block = sampler.get_H()
            syndrome = (H_block @ stego_block_bits) % 2
            extracted_message_bits.extend(syndrome.tolist())
            
    extracted_message_str = "".join(map(str, extracted_message_bits))
    extracted_message_str = extracted_message_str[:message_len]

    with open(output_path, 'w') as f:
        f.write(extracted_message_str)
        
    logger.log(f"Extracted {len(extracted_message_str)} bits.")
    logger.log(f"Extracted message saved to {output_path}")
    return extracted_message_str

def main():
    args = create_argparser().parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger.configure(dir=args.save_dir)
    
    # Calculate payload size based on the rate and truncate the message
    payload_size = int(args.sts_payload_rate * args.image_size * args.image_size)
    
    # Add a check to ensure the validity of c and h for the STS algorithm
    if args.mode in ['embed', 'extract']:
        if args.sts_constraint_height < 7 or args.sts_constraint_height > 12:
             raise ValueError(
                f"FATAL: For this implementation, the constraint height 'h' "
                f"({args.sts_constraint_height}) must be between 7 and 12."
            )

    if args.mode == 'embed':
        logger.log("Mode: Embedding with STScp algorithm")
        try:
            with open(args.message_input, 'r') as f:
                message = f.read().strip()
            if not all(c in '01' for c in message):
                raise ValueError("Message file contains non-binary characters.")
            if not message:
                raise ValueError("Message file is empty.")
            
            if len(message) < payload_size:
                logger.warn(
                    f"Original message length ({len(message)}) is less than the target payload "
                    f"({payload_size}). Using the full message. The effective payload rate will be lower."
                )
                message_to_embed = message
            else:
                message_to_embed = message[:payload_size]
            
            logger.log(f"Embedding a message of size {len(message_to_embed)} based on block length {args.sts_block_len} and payload rate {args.sts_payload_rate:g}.")

            embedded_msg_path = os.path.join(args.save_dir, f"message_embedded_{args.sts_block_len}_{args.sts_payload_rate:g}.txt")

            with open(embedded_msg_path, 'w') as f:
                f.write(message_to_embed)
            logger.log(f"Message to be embedded is saved to {embedded_msg_path}")

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error: {e}")
            return
        
        embed_message_stscp(args, message_to_embed)

    elif args.mode == 'extract':
        logger.log("Mode: Extracting with STScp algorithm")

        extracted_msg = extract_message_stscp(args)
        embedded_msg_path = os.path.join(args.save_dir, f"message_embedded_{args.sts_block_len}_{args.sts_payload_rate:g}.txt")

        if extracted_msg is None: return
        
        # Compare with the message that was actually embedded
        try:
            with open(embedded_msg_path, 'r') as f:
                original_embedded_msg = f.read().strip()
            
            if original_embedded_msg == extracted_msg:
                logger.log(f"SUCCESS: Extracted message matches the original embedded message from '{embedded_msg_path}'.")
            else:
                logger.log(f"FAILURE: Extracted message does not match the content of '{embedded_msg_path}'.")
        except FileNotFoundError:
            logger.log(f"Cannot verify message, original embedded message file '{embedded_msg_path}' not found.")
            logger.log("You can manually compare the extracted message with the original source if needed.")
    
    elif args.mode == 'generate':
        logger.log("Mode: Generating a cover image")
        generate_image(args)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main() 