import os
import json
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
from src.utils import video_processing_utils as vid_utils
from src.utils import binary_utils as bnr
from src.utils import string_utils
from src.utils import motion_vectors_utils as mv

ZIGZAG_INDICES = [
    (3, 1), (2, 2), (1, 3), (0, 4), (2, 3), (1, 4)
]


def zigzag_create_D(matrix):
    """Create a zigzag pattern from a matrix."""
    return [matrix[i, j] for i, j in ZIGZAG_INDICES]


def insert_zigzag_D(matrix, D):
    """Insert a zigzag pattern into a matrix."""
    for idx, (i, j) in enumerate(ZIGZAG_INDICES):
        matrix[i, j] = D[idx]
    return matrix


def setup_threshold_values(T, D):
    """Setup threshold values for hiding technique."""
    f = np.zeros(3)
    s = np.zeros(3)
    for u in range(3):
        if D[2*u] < 0:
            f[u] = -T
        else:
            f[u] = T
        if D[2*u+1] < 0:
            s[u] = -T
        else:
            s[u] = T
    return f, s


def hiding_technique_abs(msg_frame, f, s, D):
    """Apply the hiding technique to a coefficients."""
    S = 0
    for u in range(3):
        if S < len(msg_frame):
            if msg_frame[S] == 1:
                if abs(D[2*u]) < abs(D[2*u+1]):
                    D[2*u+1] += s[u]
                else:
                    C = D[2*u]
                    D[2*u] = D[2*u+1]
                    D[2*u+1] = C + f[u]
            else:
                if abs(D[2*u]) > abs(D[2*u+1]):
                    D[2*u] += f[u]
                else:
                    C = D[2*u]
                    D[2*u] = D[2*u+1] + s[u]
                    D[2*u+1] = C 
        S += 1
    return D

def extracting_technique_abs(D):
    """Extract the message bits from a zigzag pattern list."""
    message_bits = []
    for k in range(0, 5, 2):
        if abs(D[k]) < abs(D[k+1]):
            message_bits.append(1)
        else:
            message_bits.append(0)
    return message_bits

def encode_dtc_psyschovisual(orig_video_path, message_path, motion_blocks_file, generate_threshold=True):
    """
    Encode a message into a video using the DTC Psychovisual LSB method.
    
    Args:
        orig_video_path (str): Path to the original video.
        message_path (str): Path to the message file.
        motion_blocks_file (str): File to save the motion blocks.
        generate_threshold (bool, optional): Flag to generate threshold. Default is True.
        
    Returns:
        int: Length of the message.
    """
    message = bnr.string_to_binary_array(message_path)
    
    properties = vid_utils.video_to_rgb_frames(orig_video_path)
    motion_blocks_w_I, T_values = mv.detect_motion_blocks_and_T(properties)
    I_frame_indexes = vid_utils.get_I_frame_numbers(orig_video_path)
    motion_blocks = mv.filter_motion_blocks(motion_blocks_w_I, I_frame_indexes)
    
    mv.save_motion_blocks(motion_blocks, motion_blocks_file)
    
    T_iterator = iter(T_values)
    message_index = 0
    bits_to_hide = 3

    for frame, x, y in motion_blocks:
        if message_index >= len(message):
            break

        img_path = f"frames/frame_{frame}.png"
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = yuv_image[:, :, 0]
        block = y_channel[y:y+8, x:x+8]

        frame_message = message[message_index:message_index+bits_to_hide]
        if len(frame_message) < 3:
            frame_message = np.pad(frame_message, (0, 3 - len(frame_message)))

        dct_block = cv2.dct(np.float32(block))
        D = zigzag_create_D(dct_block)

        T = next(T_iterator, 20) if generate_threshold else 4
        f, s = setup_threshold_values(T, D)
        D = hiding_technique_abs(frame_message, f, s, D)

        dct_block = insert_zigzag_D(dct_block, D)
        modified_block = cv2.idct(dct_block)
        y_channel[y:y+8, x:x+8] = modified_block

        yuv_image[:, :, 0] = y_channel
        modified_img = cv2.cvtColor(yuv_image, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(img_path, modified_img)

        message_index += bits_to_hide

    print(f"[INFO] message encoding completed")
    vid_utils.reconstruct_video_from_rgb_frames(orig_video_path, properties)
    vid_utils.remove_dirs()

    return len(message)

def decode_dtc_psyschovisual(stego_video_path, len_b_msg, motion_blocks_file, write_file=False):
    """
    Decode a message from a video using the DTC Psychovisual LSB method.
    
    Args:
        stego_video_path (str): Path to the stego video.
        len_b_msg (int): Length of the binary message.
        motion_blocks_file (str): File containing the motion blocks.
        write_file (bool, optional): Flag to write decoded message to a file. Default is False.
        
    Returns:
        str: Decoded message.
    """
    properties = vid_utils.video_to_rgb_frames(stego_video_path)
    message = []
    motion_blocks = mv.load_motion_blocks(motion_blocks_file)

    for frame, x, y in motion_blocks:
        if len(message) >= len_b_msg:
            break

        img_path = f"frames/frame_{frame}.png"
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = yuv_image[:, :, 0]

        block = y_channel[y:y+8, x:x+8]
        dct_block = cv2.dct(np.float32(block))
        D = zigzag_create_D(dct_block)
        three_decoded_bits = extracting_technique_abs(D)

        for bit in three_decoded_bits:
            if len(message) < len_b_msg:
                message.append(bit)
            else:
                break
            
    decoded_message = bnr.binary_array_to_string(np.array(message))

    if write_file:
        string_utils.write_message_to_file(decoded_message, "decoded_message.txt")
        print("[INFO] Saved decoded message as decoded_message.txt")

    vid_utils.remove_dirs()
    return decoded_message