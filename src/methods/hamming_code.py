import cv2 as cv
import numpy as np

from src.utils import binary_utils as bnr
from src.utils import video_processing_utils as vid_utils
from src.utils import string_utils

# Generator matrix
G = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1]
])

# Parity-check matrix (transposed)
H_TRANSPOSED = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 1],
    [1, 0, 1]
])


def hamming_encode(orig_video_path, message_path, shift_key, col_key, row_key, xor_key):
    """
    Encode a message into a video using Hamming Code (7, 4).

    Args:
        orig_video_path (str): Path to the original video.
        message_path (str): Path to the message file.
        shift_key (int): Key for shifting the message.
        col_key (int): Key for column selection.
        row_key (int): Key for row selection.
        xor_key (np.array): Key for XOR operation.

    Returns:
        int: Length of the embedded message.
    """
    vid_properties = vid_utils.video_to_rgb_frames(orig_video_path)
    vid_utils.create_dirs()

    for i in range(1, int(vid_properties["frames"]) + 1):
        vid_utils.rgb2yuv(f"frame_{i}.png")

    max_codew_p_frame = int(vid_properties["height"] * vid_properties["width"] * 0.22)

    message = bnr.string_to_binary_array(message_path)
    message = bnr.fill_end_zeros(np.roll(message, shift_key), 4)

    codew_p_frame, codew_p_last_frame = vid_utils.distribution_of_bits_between_frames(
        len(message), vid_properties["frames"], 4
    )

    if codew_p_frame > max_codew_p_frame:
        print("[INFO] Message is too large to embed")
        return

    actual_max_codew = 1 if codew_p_frame == 0 else codew_p_frame

    row = 0
    col = 0
    embedded_codewords_per_frame = 0
    curr_frame = 1

    for i in range(0, len(message), 4):
        four_bits = message[i:i + 4]
        codeword = hamming_encode_codeword(four_bits)
        codeword = codeword ^ xor_key

        if embedded_codewords_per_frame == 0:
            y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
            u_component_path = f"./tmp/U/frame_{curr_frame}.png"
            v_component_path = f"./tmp/V/frame_{curr_frame}.png"

            y_frame = cv.imread(y_component_path, cv.IMREAD_GRAYSCALE)
            u_frame = cv.imread(u_component_path, cv.IMREAD_GRAYSCALE)
            v_frame = cv.imread(v_component_path, cv.IMREAD_GRAYSCALE)

            row = 0
            col = 0

        y_binary_value = format(y_frame[row, col], '#010b')
        u_binary_value = format(u_frame[row, col], '#010b')
        v_binary_value = format(v_frame[row, col], '#010b')

        y_frame[row, col] = int(y_binary_value[:-3] + ''.join(str(bit) for bit in codeword[:3]), 2)
        u_frame[row, col] = int(u_binary_value[:-2] + ''.join(str(bit) for bit in codeword[3:5]), 2)
        v_frame[row, col] = int(v_binary_value[:-2] + ''.join(str(bit) for bit in codeword[5:]), 2)

        embedded_codewords_per_frame += 1

        col += 1
        if col >= int(vid_properties["width"]):
            col = 0
            row += 1

        if embedded_codewords_per_frame >= actual_max_codew:
            curr_frame += 1
            embedded_codewords_per_frame = 0

            cv.imwrite(y_component_path, y_frame)
            cv.imwrite(u_component_path, u_frame)
            cv.imwrite(v_component_path, v_frame)

            if curr_frame == vid_properties["frames"]:
                actual_max_codew = codew_p_last_frame

    if embedded_codewords_per_frame > 0:
        cv.imwrite(y_component_path, y_frame)
        cv.imwrite(u_component_path, u_frame)
        cv.imwrite(v_component_path, v_frame)

    print("[INFO] message is encoded to frames")

    for i in range(1, int(vid_properties["frames"]) + 1):
        vid_utils.yuv2rgb(f"frame_{i}.png")

    vid_utils.reconstruct_video_from_rgb_frames(orig_video_path, vid_properties)
    vid_utils.remove_dirs()

    print("[INFO] embedding finished")
    return len(message)


def hamming_decode(stego_video_path, shift_key, col_key, row_key, message_len, xor_key, write_file=False):
    """
    Decode a message from a steganographic video.

    Args:
        stego_video_path (str): Path to the steganographic video.
        shift_key (int): Key for shifting the message.
        col_key (int): Key for column selection.
        row_key (int): Key for row selection.
        message_len (int): Length of the embedded message.
        xor_key (np.array): Key for XOR operation.
        write_file (bool): Whether to write the decoded message to a file.

    Returns:
        str: Decoded message.
    """
    decoded_message = []

    codeword_chaos = np.array([0, 0, 0, 0, 0, 0, 0])
    decoded_codeword = np.array([0, 0, 0, 0])

    # Convert the video stream into frames. Separate each frame into Y, U and V components.
    vid_properties = vid_utils.video_to_rgb_frames(stego_video_path)
    vid_utils.create_dirs()

    for i in range(1, int(vid_properties["frames"]) + 1):
        vid_utils.rgb2yuv(f"frame_{i}.png")

    codew_p_frame, codew_p_last_frame = vid_utils.distribution_of_bits_between_frames(
        message_len, vid_properties["frames"], 4
    )

    actual_max_codew = 1 if codew_p_frame == 0 else codew_p_frame

    for curr_frame in range(1, int(vid_properties["frames"]) + 1):
        embedded_codewords = 0

        y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
        u_component_path = f"./tmp/U/frame_{curr_frame}.png"
        v_component_path = f"./tmp/V/frame_{curr_frame}.png"

        y_frame = cv.imread(y_component_path, cv.IMREAD_GRAYSCALE)
        u_frame = cv.imread(u_component_path, cv.IMREAD_GRAYSCALE)
        v_frame = cv.imread(v_component_path, cv.IMREAD_GRAYSCALE)

        if curr_frame == vid_properties["frames"]:
            actual_max_codew = codew_p_last_frame

        stop_loop = False
        for row in range(int(vid_properties["height"])):
            if stop_loop:
                break

            for col in range(int(vid_properties["width"])):
                if embedded_codewords >= actual_max_codew:
                    stop_loop = True
                    break

                y_binary_value = format(y_frame[row, col], '#010b')
                u_binary_value = format(u_frame[row, col], '#010b')
                v_binary_value = format(v_frame[row, col], '#010b')

                codeword_chaos[0] = y_binary_value[-3]
                codeword_chaos[1] = y_binary_value[-2]
                codeword_chaos[2] = y_binary_value[-1]

                codeword_chaos[3] = u_binary_value[-2]
                codeword_chaos[4] = u_binary_value[-1]

                codeword_chaos[5] = v_binary_value[-2]
                codeword_chaos[6] = v_binary_value[-1]

                codeword = codeword_chaos ^ xor_key
                decoded_codeword = hamming_decode_codeword(codeword)
                decoded_message.extend(decoded_codeword)

                embedded_codewords += 1

        if codew_p_frame == 0 and codew_p_last_frame == curr_frame:
            break

    output_message = np.roll(np.array(decoded_message), -shift_key)
    message = bnr.binary_array_to_string(output_message)

    if write_file:
        string_utils.write_message_to_file(message, "decoded_message.txt")
        print("[INFO] Saved decoded message as decoded_message.txt")

    vid_utils.remove_dirs()
    return message


def hamming_encode_codeword(four_bits):
    """Encode 4 bits using Hamming Code (7, 4)."""
    return np.dot(four_bits, G) % 2


def hamming_decode_codeword(codeword):
    """Decode a 7-bit Hamming Code (7, 4) codeword."""
    Z = np.dot(codeword, H_TRANSPOSED) % 2
    R = codeword

    index = -1
    for i, H_row in enumerate(H_TRANSPOSED):
        if np.all(Z == H_row):
            index = i

    if index > -1:
        R[index] = 1 - R[index]

    return R[-4:]


def fill_end_zeros(input_array):
    """Pad the input array with zeros to make its length a multiple of 4."""
    return np.pad(input_array, (0, -len(input_array) % 4), mode='constant')