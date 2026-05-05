import numpy as np
import scipy
import torch
from guided_diffusion.image_datasets import load_data
from typing import Tuple, List
from guided_diffusion import logger

def load_reference(data_dir, batch_size, image_size, class_cond=False):
    # data_dir= ref_imgs/face
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs
    
def _get_denormalized_pixel_params(normalized_mu_flat: np.ndarray, normalized_sigma_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denormalizes flat arrays of mu and sigma from [-1, 1] range to [0, 255] pixel space.
    Ensures sigma is positive and has a minimum value.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    normalized_mu_flat_np = np.asarray(normalized_mu_flat)
    normalized_sigma_flat_np = np.asarray(normalized_sigma_flat)

    # Denormalize from [-1, 1] to [0, 255] for mean
    denormalized_mu = (normalized_mu_flat_np + 1.0) / 2.0 * 255.0
    
    # Scale sigma from normalized space to 0-255 space
    # If sigma in [-1,1] represented std dev in that space,
    # then in [0,255] space, the range is 255, so std dev scales by 255/2
    denormalized_sigma = normalized_sigma_flat_np * (255.0 / 2.0)
    
    # Ensure sigma is positive (it should be if it's a std dev from model.learn_sigma=True)
    # Taking absolute value is a robust way if underlying model can sometimes output negative sigma.
    denormalized_sigma = np.abs(denormalized_sigma)
    
    # Set a minimum sigma to avoid division by zero or overly sharp distributions
    epsilon = 1e-3 # A small standard deviation in pixel space
    denormalized_sigma = np.maximum(denormalized_sigma, epsilon)
    
    return denormalized_mu, denormalized_sigma

def get_probs_indices_from_diffu(
    mu_flat_norm: np.ndarray, 
    sigma_flat_norm: np.ndarray, 
    num_channels: int, 
    height: int, 
    width: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Calculates the probability distribution (0-255) for each pixel based on
    normalized mu and sigma parameters from the diffusion model.

    Args:
        mu_flat_norm: Flattened array of normalized means for all pixels/channels.
        sigma_flat_norm: Flattened array of normalized sigmas for all pixels/channels.
        num_channels: Number of image channels (e.g., 3 for RGB).
        height: Image height.
        width: Image width.

    Returns:
        Tuple (indices_map_list, probs_map_list):
            indices_map_list: List (per channel) of np.arrays (pixels_per_channel, 256),
                              where each row is np.arange(256).
            probs_map_list: List (per channel) of np.arrays (pixels_per_channel, 256),
                            containing the probabilities for each pixel value 0-255.
    """
    logger.log("Getting probs and indices from diffusion model...")
    denormalized_mus_flat, denormalized_sigmas_flat = _get_denormalized_pixel_params(
        mu_flat_norm, sigma_flat_norm
    )

    num_pixels_per_channel = height * width
    
    # Reshape flat params to (num_channels, num_pixels_per_channel)
    # Assuming mu_flat_norm and sigma_flat_norm are ordered C,H,W when flattened
    try:
        mus_reshaped = denormalized_mus_flat.reshape((num_channels, num_pixels_per_channel))
        sigmas_reshaped = denormalized_sigmas_flat.reshape((num_channels, num_pixels_per_channel))
    except ValueError as e:
        expected_len = num_channels * num_pixels_per_channel
        raise ValueError(
            f"Error reshaping mu/sigma. Expected flat length {expected_len}, "
            f"got mu: {denormalized_mus_flat.size}, sigma: {denormalized_sigmas_flat.size}. Error: {e}"
        )


    all_pixel_values = np.arange(256, dtype=float) # Values k = 0, 1, ..., 255
    cdf_upper_bounds = all_pixel_values + 0.5
    cdf_lower_bounds = all_pixel_values - 0.5

    indices_map_list = []
    probs_map_list = []

    for c_idx in range(num_channels):
        channel_mus = mus_reshaped[c_idx, :]      # Shape (num_pixels_per_channel,)
        channel_sigmas = sigmas_reshaped[c_idx, :] # Shape (num_pixels_per_channel,)
        
        # Vectorized calculation for all pixels in the channel
        # Reshape mus and sigmas to (n, 1) to broadcast against bounds of shape (256,)
        # The result of cdf will be (num_pixels_per_channel, 256)
        mus_col = channel_mus[:, np.newaxis]
        sigmas_col = channel_sigmas[:, np.newaxis]

        probs_at_upper = scipy.stats.norm(loc=mus_col, scale=sigmas_col).cdf(cdf_upper_bounds)
        probs_at_lower = scipy.stats.norm(loc=mus_col, scale=sigmas_col).cdf(cdf_lower_bounds)
        
        current_channel_probs = probs_at_upper - probs_at_lower
        current_channel_probs[current_channel_probs < 0] = 0 # Ensure non-negativity

        # Normalize probabilities for each pixel to sum to 1
        sum_probs = np.sum(current_channel_probs, axis=1, keepdims=True)
        
        # Avoid division by zero
        # Create a mask for rows with a sum of probabilities greater than a small threshold
        valid_mask = sum_probs > 1e-7
        
        # Initialize with a fallback distribution for rows with sum_probs near zero
        fallback_probs = np.zeros_like(current_channel_probs)
        clipped_mus_int = np.round(np.clip(channel_mus, 0, 255)).astype(int)
        fallback_probs[np.arange(len(fallback_probs)), clipped_mus_int] = 1.0
        
        # Apply fallback where sum_probs is too small
        current_channel_probs = np.where(valid_mask, current_channel_probs / sum_probs, fallback_probs)

        # Create indices map for the current channel
        num_pixels_per_channel = mus_reshaped.shape[1]
        current_channel_indices = np.tile(all_pixel_values, (num_pixels_per_channel, 1))
            
        indices_map_list.append(current_channel_indices)
        probs_map_list.append(current_channel_probs)
        
    logger.log("Getted probs and indices from diffusion model")    
    return indices_map_list, probs_map_list

def round_pix(ten):
    '''ten:tensor类型'''
    ten = ten.squeeze(0)
    ten = (ten + 1.) / 2
    ten = ten.mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu",torch.uint8).numpy()
    return ten

def probs_indices_filter(indices, probs, top_p):
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    if not isinstance(probs, np.ndarray):
        probs = np.array(probs)
    diff = np.diff(indices)
    bound_index = np.where(diff < 0)[0][0] + 1 if np.any(diff < 0) else 0
    if bound_index < indices.size / 2:
        filtered_indices = indices[bound_index:]
        filtered_probs = probs[bound_index:]
    else:
        filtered_indices = indices[:bound_index]
        filtered_probs = probs[:bound_index]
    sort_indices = np.argsort(filtered_probs)[::-1]
    final_indices = filtered_indices[sort_indices]
    sorted_probs = filtered_probs[sort_indices]
    final_probs = sorted_probs / sorted_probs.sum()
    if not (top_p is None or top_p == 1.0):
        cum_probs = final_probs.cumsum(0)
        k = np.argmax(cum_probs > top_p) + 1
        final_probs = final_probs[:k]
        final_indices = final_indices[:k]
        final_probs = 1 / cum_probs[k - 1] * final_probs  # Normalizing
   
    return final_indices, final_probs

def get_zero_tff_qua_pix_probability_numpy(m: list, b: list, qua_pix: list, mode: int):
    '''加快计算速度
        m:一个通道的所有均值，list
        b:一个通道的所有方差，list
        qua_pix:元素值
    '''
    len_m = len(m)
    cdf_add_list = np.array([0.5] * len_m) + np.array(qua_pix) + np.array(
        [mode] * len_m)
    cdf_minus_list = np.array([-0.5] * len_m) + np.array(qua_pix) + np.array(
        [mode] * len_m)
    pro = scipy.stats.norm(m, b).cdf(cdf_add_list.tolist()) - scipy.stats.norm(
        m, b).cdf(cdf_minus_list.tolist())
    return pro

class ImageSettings:
    def __init__(self, algo: str = 'arithmetic', temp: float = 1.0, top_p: float = 1.0, seed: int = 100, image_size: int = 256):
        self.algo = algo
        self.temp = temp
        self.top_p = top_p
        self.seed = seed
        self.image_size = image_size

    def __call__(self):
        return self.algo, self.temp, self.top_p, self.seed, self.image_size

def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i

def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]