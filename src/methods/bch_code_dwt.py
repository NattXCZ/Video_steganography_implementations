import numpy as np
import cv2
import pywt
from math import floor

from src.utils import video_processing_utils as vid_utils
from src.utils import binary_utils as bnr
from galois import BCH

FLOAT_PAIRS = (
    (3.60, 0.65), (3.61, 0.64), (3.59, 0.66), (3.62, 0.63),
    (3.58, 0.67), (3.63, 0.62), (3.57, 0.68), (3.64, 0.61)
)

def get_positions(col_key, row_key, width, height):
    np.random.seed(col_key + row_key)
    # Použijeme poloviční rozměry, protože DWT zmenšuje obrázek na polovinu
    positions = [(i, j) for i in range(height//2) for j in range(width//2)]
    np.random.shuffle(positions)
    return positions

def encode_bch_dwt(orig_video_path, message_path, xor_key, col_key, row_key, bch_num=11):
    """
    Encode a message into a video using BCH coding and DWT.

    Args:
        orig_video_path (str): Path to the original video.
        message_path (str): Path to the message file.
        xor_key (np.array): XOR key for encoding.
        bch_num (int): BCH parameter. Default is 11.
 
    Returns:
        tuple: Number of codewords per frame and in the last frame.
    """
    bch = BCH(15, bch_num)
    
    bin_arr = bnr.string_to_binary_array(message_path)
    chaos_arr = process_binary_array(bin_arr)
    chaos_arr = bnr.fill_end_zeros(chaos_arr, bch_num)
    message_len = len(chaos_arr)

    vid_properties = vid_utils.video_to_rgb_frames(orig_video_path)
    vid_utils.create_dirs()
    
    for i in range(1, int(vid_properties["frames"]) + 1):
        vid_utils.rgb2yuv(f"frame_{i}.png")
    
    codew_p_frame, codew_p_last_frame = vid_utils.distribution_of_bits_between_frames(
        message_len, vid_properties["frames"], bch_num
    )
    
    actual_max_codew = 1 if codew_p_frame == 0 else codew_p_frame
    
    embedded_codewords_per_frame = 0
    curr_frame = 1

    y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
    y_frame = cv2.imread(y_component_path, cv2.IMREAD_GRAYSCALE)
    height, width = y_frame.shape

    positions = get_positions(col_key, row_key, width, height)
    used_positions = set()
    
    for i in range(0, len(chaos_arr), bch_num):
        msg_bits = chaos_arr[i:i+bch_num]
        codeword = bch.encode(msg_bits) ^ xor_key

        if embedded_codewords_per_frame == 0:
            y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
            u_component_path = f"./tmp/U/frame_{curr_frame}.png"
            v_component_path = f"./tmp/V/frame_{curr_frame}.png"

            y_frame = cv2.imread(y_component_path, cv2.IMREAD_GRAYSCALE)
            u_frame = cv2.imread(u_component_path, cv2.IMREAD_GRAYSCALE)
            v_frame = cv2.imread(v_component_path, cv2.IMREAD_GRAYSCALE)

            coeffs_Y = pywt.dwt2(y_frame, 'haar')
            coeffs_U = pywt.dwt2(u_frame, 'haar')
            coeffs_V = pywt.dwt2(v_frame, 'haar')
            
            LL_Y, (LH_Y, HL_Y, HH_Y) = coeffs_Y
            LL_U, (LH_U, HL_U, HH_U) = coeffs_U
            LL_V, (LH_V, HL_V, HH_V) = coeffs_V

        while True:
            if not positions:
                raise ValueError("Ran out of positions to embed data")
            row, col = positions.pop(0)
            if (row, col) not in used_positions:
                used_positions.add((row, col))
                break
        
        LH_Y_bin = format(floor(abs(LH_Y[row, col] + 0.0000001)), '#010b')
        HL_Y_bin = format(floor(abs(HL_Y[row, col] + 0.0000001)), '#010b')
        HH_Y_bin = format(floor(abs(HH_Y[row, col] + 0.0000001)), '#010b')
        
        LH_U_bin = format(floor(abs(LH_U[row, col] + 0.0000001)), '#010b')
        HL_U_bin = format(floor(abs(HL_U[row, col] + 0.0000001)), '#010b')
        HH_U_bin = format(floor(abs(HH_U[row, col] + 0.0000001)), '#010b')
        
        LH_V_bin = format(floor(abs(LH_V[row, col] + 0.0000001)), '#010b')
        HL_V_bin = format(floor(abs(HL_V[row, col] + 0.0000001)), '#010b')
        HH_V_bin = format(floor(abs(HH_V[row, col] + 0.0000001)), '#010b')
        
        LH_Y[row, col] = int(LH_Y_bin[:-3] + ''.join(str(bit) for bit in codeword[:3]), 2)
        HL_Y[row, col] = int(HL_Y_bin[:-3] + ''.join(str(bit) for bit in codeword[3:6]), 2)
        HH_Y[row, col] = int(HH_Y_bin[:-3] + ''.join(str(bit) for bit in codeword[6:9]), 2)
        
        LH_U[row, col] = int(LH_U_bin[:-1] + str(codeword[9]), 2)
        HL_U[row, col] = int(HL_U_bin[:-1] + str(codeword[10]), 2)
        HH_U[row, col] = int(HH_U_bin[:-1] + str(codeword[11]), 2)
        
        LH_V[row, col] = int(LH_V_bin[:-1] + str(codeword[12]), 2)
        HL_V[row, col] = int(HL_V_bin[:-1] + str(codeword[13]), 2)
        HH_V[row, col] = int(HH_V_bin[:-1] + str(codeword[14]), 2)

        embedded_codewords_per_frame += 1 

        if embedded_codewords_per_frame >= actual_max_codew:
            embedded_codewords_per_frame = 0
            
            HH_Y = HH_Y.reshape(LL_Y.shape)
            HL_Y = HL_Y.reshape(LL_Y.shape)
            LH_Y = LH_Y.reshape(LL_Y.shape)
            
            HH_U = HH_U.reshape(LL_U.shape)
            HL_U = HL_U.reshape(LL_U.shape)
            LH_U = LH_U.reshape(LL_U.shape)
            
            HH_V = HH_V.reshape(LL_V.shape)
            HL_V = HL_V.reshape(LL_V.shape)
            LH_V = LH_V.reshape(LL_V.shape)
            
            Y_new = pywt.idwt2((LL_Y, (LH_Y, HL_Y, HH_Y)), 'haar')
            U_new = pywt.idwt2((LL_U, (LH_U, HL_U, HH_U)), 'haar')
            V_new = pywt.idwt2((LL_V, (LH_V, HL_V, HH_V)), 'haar')
    
            cv2.imwrite(y_component_path, Y_new)
            cv2.imwrite(u_component_path, U_new)
            cv2.imwrite(v_component_path, V_new)
            
            curr_frame += 1
            if curr_frame == vid_properties["frames"]:
                actual_max_codew = codew_p_last_frame

    print("[INFO] message is encoded to frames")
    
    for i in range(1, int(vid_properties["frames"]) + 1):
        vid_utils.yuv2rgb(f"frame_{i}.png")
        
    vid_utils.reconstruct_video_from_rgb_frames(orig_video_path, vid_properties)
    vid_utils.remove_dirs()
    
    print("[INFO] embedding finished")
    
    return message_len

def decode_bch_dwt(stego_video_path, message_len, xor_key, col_key, row_key, bch_num=11, write_file=False):
    """
    Decode a message from a video using BCH coding and DWT.

    Args:
        stego_video_path (str): Path to the stego video.
        codew_p_frame (int): Number of codewords per frame.
        codew_p_last_frame (int): Number of codewords in the last frame.
        xor_key (np.array): XOR key for decoding.
        bch_num (int): BCH parameter. Default is 11.
        write_file (bool): Whether to write the decoded message to a file.

    Returns:
        str: Decoded message.
    """
    bch = BCH(15, bch_num)
    decoded_message = []
    codeword_chaos =  np.zeros(15, dtype = np.uint8)
    decoded_codeword = np.zeros(bch_num, dtype = np.uint8)
 
    vid_properties = vid_utils.video_to_rgb_frames(stego_video_path)
    vid_utils.create_dirs()
    
    codew_p_frame, codew_p_last_frame = vid_utils.distribution_of_bits_between_frames(
        message_len, vid_properties["frames"], bch_num
    )

    actual_max_codew = 1 if codew_p_frame == 0 else codew_p_frame
   
    for i in range(1, int(vid_properties["frames"]) + 1):
        vid_utils.rgb2yuv(f"frame_{i}.png")

    y_component_path = f"./tmp/Y/frame_1.png"
    y_frame = cv2.imread(y_component_path, cv2.IMREAD_GRAYSCALE)
    height, width = y_frame.shape

    positions = get_positions(col_key, row_key, width, height)
    used_positions = set()

    for curr_frame in range(1, int(vid_properties["frames"]) + 1):
        y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
        u_component_path = f"./tmp/U/frame_{curr_frame}.png"
        v_component_path = f"./tmp/V/frame_{curr_frame}.png"

        y_frame = cv2.imread(y_component_path, cv2.IMREAD_GRAYSCALE)
        u_frame = cv2.imread(u_component_path, cv2.IMREAD_GRAYSCALE)
        v_frame = cv2.imread(v_component_path, cv2.IMREAD_GRAYSCALE)
        
        coeffs_Y = pywt.dwt2(y_frame, 'haar')
        coeffs_U = pywt.dwt2(u_frame, 'haar')
        coeffs_V = pywt.dwt2(v_frame, 'haar')
            
        LL_Y, (LH_Y, HL_Y, HH_Y) = coeffs_Y
        LL_U, (LH_U, HL_U, HH_U) = coeffs_U
        LL_V, (LH_V, HL_V, HH_V) = coeffs_V
        
        if curr_frame == vid_properties["frames"]:
            actual_max_codew = codew_p_last_frame

        embedded_codewords = 0

        while embedded_codewords < actual_max_codew and positions:
            row, col = positions.pop(0)
            if (row, col) in used_positions:
                continue
            used_positions.add((row, col))

            LH_Y_bin = format(floor(abs(LH_Y[row, col] + 0.0000001)), '#010b')
            HL_Y_bin = format(floor(abs(HL_Y[row, col] + 0.0000001)), '#010b')
            HH_Y_bin = format(floor(abs(HH_Y[row, col] + 0.0000001)), '#010b')
    
            LH_U_bin = format(floor(abs(LH_U[row, col] + 0.0000001)), '#010b')
            HL_U_bin = format(floor(abs(HL_U[row, col] + 0.0000001)), '#010b')
            HH_U_bin = format(floor(abs(HH_U[row, col] + 0.0000001)), '#010b')
    
            LH_V_bin = format(floor(abs(LH_V[row, col] + 0.0000001)), '#010b')
            HL_V_bin = format(floor(abs(HL_V[row, col] + 0.0000001)), '#010b')
            HH_V_bin = format(floor(abs(HH_V[row, col] + 0.0000001)), '#010b')

            codeword_chaos[0] = LH_Y_bin[-3]
            codeword_chaos[1] = LH_Y_bin[-2]
            codeword_chaos[2] = LH_Y_bin[-1]
                
            codeword_chaos[3] = HL_Y_bin[-3]
            codeword_chaos[4] = HL_Y_bin[-2]
            codeword_chaos[5] = HL_Y_bin[-1]
                
            codeword_chaos[6] = HH_Y_bin[-3]
            codeword_chaos[7] = HH_Y_bin[-2]
            codeword_chaos[8] = HH_Y_bin[-1]
    
            codeword_chaos[9] = LH_U_bin[-1]
            codeword_chaos[10] = HL_U_bin[-1]
            codeword_chaos[11] = HH_U_bin[-1]
                
            codeword_chaos[12] = LH_V_bin[-1]
            codeword_chaos[13] = HL_V_bin[-1]
            codeword_chaos[14] = HH_V_bin[-1]
            
            codeword = codeword_chaos ^ xor_key
            decoded_codeword = bch.decode(codeword)
            decoded_message.extend(decoded_codeword)
            
            embedded_codewords += 1
            
        if codew_p_frame == 0 and codew_p_last_frame == curr_frame:
            break

    output_message = process_binary_array(np.array(decoded_message), False)
    message = bnr.binary_array_to_string(output_message)
    
    if write_file:
        with open("decoded_message.txt", 'w', encoding='utf-8') as file:
            file.write(message)
        print("[INFO] saved decoded message as decoded_message.txt")

    vid_utils.remove_dirs()
    return message

def logistic_key(N, mu, x0):
    """Generate a logistic key sequence."""
    X = np.zeros(N)
    X[0] = mu * x0 * (1 - x0) 
    for k in range(1, N):
        X[k] = mu * X[k-1] * (1 - X[k-1])
    return X

def create_B(logistic_seq):
    """Create a binary sequence from a logistic sequence."""
    T = np.mean(logistic_seq)
    return np.array([1 if x >= T else 0 for x in logistic_seq], dtype=np.uint8)

def process_part_component(binary_array, chaos_array, start_index, end_index, B):
    """Process a part of the binary array using chaos algorithm and logistic mapping."""
    k = 0
    for i in range(start_index, end_index):
        C = binary_array[i]
        B_val = B[k]
        chaos_array[i] = C ^ B_val
        k += 1
            
    return chaos_array

def process_binary_array(binary_array, encode = True):
    """Process a binary array using logistic mapping."""
    original_length = len(binary_array)
    
    if encode:
        if original_length % 8 != 0:
            padding_length = 8 - (original_length % 8)
            binary_array = np.append(binary_array, np.zeros(padding_length, dtype=np.uint8))
    else:
        len_8 = (original_length // 8) * 8
        if len_8 != original_length:
            diff = original_length - len_8
            binary_array = binary_array[:-diff]
            
    segment_length = len(binary_array) // 8
    chaos_array = np.zeros_like(binary_array, dtype=np.uint8)
    
    i = 0
    for float_pair in FLOAT_PAIRS:
        start_index = i * segment_length
        end_index = start_index + segment_length
        
        generatedKey = logistic_key(segment_length, float_pair[0], float_pair[1])
        process_part_component(binary_array, chaos_array, start_index, end_index, create_B(generatedKey))
        i += 1

    return chaos_array

