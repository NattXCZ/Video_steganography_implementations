import os

import json
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim


def filter_motion_blocks(motion_blocks, remove_frame):
    """Filters a list of tuples (frame, x, y) and returns a new list that excludes tuples with frames present in remove_frame."""
    return [(frame, x, y) for frame, x, y in motion_blocks if frame not in remove_frame]


def save_motion_blocks(motion_blocks, filename):
    """Save motion blocks to a file."""
    with open(filename, 'w') as f:
        json.dump(motion_blocks, f)


def load_motion_blocks(filename):
    """Load motion blocks from a file."""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_ssim(img1, img2):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim(img1_gray, img2_gray, full=True)
    return ssim_value


def calculate_nc(img1, img2):
    """Calculate the Normalized Correlation (NC) between two images."""
    img1_norm = cv2.normalize(img1.astype(
        'float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img2_norm = cv2.normalize(img2.astype(
        'float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return np.mean(img1_norm * img2_norm)


def determine_threshold(ssim_value, nc_value, alpha=0.5):
    """Determine the threshold value based on SSIM and NC values. """
    return alpha * ssim_value + (1 - alpha) * nc_value


def detect_motion_blocks_and_T(properties):
    """Detect motion blocks and calculate threshold values for each frame."""
    motion_blocks = []
    T_values = []
    prev_frame = None

    for frame in range(int(properties['frames'])):
        if os.path.exists(f"frames/frame_{frame}.png"):
            img_path = f"frames/frame_{frame}.png"
        else:
            continue

        current_frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if current_frame is None:
            continue

        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            for i in range(0, current_frame.shape[0], 8):
                for j in range(0, current_frame.shape[1], 8):
                    block_magnitude = magnitude[i:i+8, j:j+8]
                    avg_magnitude = np.mean(block_magnitude)
                    if avg_magnitude >= 7:
                        motion_blocks.append((frame, j, i))

            ssim_value = ssim(prev_frame, current_frame)
            nc_value = calculate_nc(prev_frame, current_frame)
            T = determine_threshold(ssim_value, nc_value)
            T_values.append(T)

        prev_frame = current_frame

    return motion_blocks, T_values
