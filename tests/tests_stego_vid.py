import os

import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
from src.utils.video_processing_utils import (
    video_to_rgb_frames,
    reconstruct_video_from_rgb_frames,
    create_dirs,
    remove_dirs,
)


def load_video(file_path):
    """Loads a video file and returns it as a numpy array."""
    create_dirs()
    properties = video_to_rgb_frames(file_path)
    frames = []

    for filename in sorted(os.listdir("./frames")):
        if filename.endswith(".png"):
            frame = cv2.imread(os.path.join("./frames", filename))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    return np.array(frames), properties


def save_video(frames, output_file, properties):
    """Saves a numpy array of frames as a video file."""
    for i, frame in enumerate(frames):
        cv2.imwrite(f"./frames/frame_{i+1}.png",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    reconstruct_video_from_rgb_frames(output_file, properties)
    remove_dirs()


def calculate_mse_psnr(original, stego):
    """Calculates Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) between original and stego videos."""
    mse = np.mean((original - stego) ** 2)

    if mse == 0:
        return 0, float("inf")

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return mse, psnr


def calculate_ssim(original, stego):
    """Calculates Structural Similarity Index (SSIM) between original and stego videos."""
    if original.shape != stego.shape:
        raise ValueError(
            "Original and stego videos must have the same dimensions")

    ssim_values = []

    for orig_frame, stego_frame in zip(original, stego):
        min_dim = min(orig_frame.shape[0], orig_frame.shape[1])
        win_size = min_dim if min_dim % 2 != 0 else min_dim - 1
        ssim_frame = ssim(orig_frame, stego_frame,
                          win_size=win_size, channel_axis=2)
        ssim_values.append(ssim_frame)

    return np.mean(ssim_values)


def calculate_ber(original_bits, received_bits):
    """Calculates Bit Error Rate (BER) between original and received bit sequences."""
    if len(original_bits) != len(received_bits):
        raise ValueError(
            "Original and received bit sequences must be of the same length."
        )

    error_bits = np.sum(original_bits != received_bits)
    total_bits = len(original_bits)
    return error_bits / total_bits


def calculate_embedding_capacity(stego, secret_data_size):
    """
    Calculates the embedding capacity of a stego video."""
    total_elements = stego.size
    capacity = np.log2(secret_data_size)
    relative_capacity = capacity / total_elements
    return capacity, relative_capacity


def calculate_bitrate(file_path):
    """Calculates the bitrate of a video file."""
    file_size = os.path.getsize(file_path) * 8  # size in bits
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frame_count / fps
    return file_size / duration


def calculate_bitrate_increase(original_file, stego_file):
    """Calculates the increase in bitrate between the original and stego videos."""
    original_bitrate = calculate_bitrate(original_file)
    stego_bitrate = calculate_bitrate(stego_file)
    return (stego_bitrate - original_bitrate) / original_bitrate


def run_steganography_tests(
    original_file, stego_file, msg_orig_bits, msr_received_bits
):
    """Runs a series of steganography tests on original and stego videos."""
    original_video, original_properties = load_video(original_file)
    stego_video, stego_properties = load_video(stego_file)

    results = {}

    # MSE and PSNR
    mse, psnr = calculate_mse_psnr(original_video, stego_video)
    results["MSE"] = mse
    results["PSNR"] = psnr

    # SSIM
    ssim_value = calculate_ssim(original_video, stego_video)
    results["SSIM"] = ssim_value

    # BER
    ber = calculate_ber(msg_orig_bits, msr_received_bits)
    results["BER"] = ber

    # Bitrate Increase
    bitrate_increase = calculate_bitrate_increase(original_file, stego_file)
    results["Bitrate Increase"] = bitrate_increase

    return results


# Example usage
if __name__ == "__main__":
    original_file = r"data_testing/bunny.avi"
    stego_file = "video.avi"
    msg_orig_bits = []
    msr_received_bits = []
    results = run_steganography_tests(
        original_file, stego_file, msg_orig_bits, msr_received_bits
    )

    for metric, value in results.items():
        print(f"{metric}: {value}")
