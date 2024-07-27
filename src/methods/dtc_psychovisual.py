import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from src.utils import video_processing_utils as vid_utils
from src.utils import binary_utils as bnr
from src.utils import string_utils


#!uklada vektory

import json

def save_motion_blocks(motion_blocks, filename):
    with open(filename, 'w') as f:
        json.dump(motion_blocks, f)

def load_motion_blocks(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
    
    

#uplne supr cupr metoda
zigzag_indices = [
    (3, 1), (2, 2), (1, 3), (0, 4), (2, 3), (1, 4)
]

def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim(img1_gray, img2_gray, full=True)
    return ssim_value

def calculate_nc(img1, img2):
    img1_norm = cv2.normalize(img1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img2_norm = cv2.normalize(img2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    nc_value = np.mean(img1_norm * img2_norm)
    return nc_value

def determine_threshold(ssim_value, nc_value, alpha=0.5):
    T = alpha * ssim_value + (1 - alpha) * nc_value
    return T

def zigzag_create_D(matrix):
    D = [matrix[i, j] for i, j in zigzag_indices]
    return D

def insert_zigzag_D(matrix, D):
    for idx, (i, j) in enumerate(zigzag_indices):
        matrix[i, j] = D[idx]
    return matrix

def setup_threshold_values(T, D):
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

def hiding_technique_abs(frame, f, s, D):
    S = 0
    for u in range(3):
        if S < len(frame):
            if frame[S] == 1:
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
    message_bits = []
    for k in range(0, 5, 2):
        if abs(D[k]) < abs(D[k+1]):
            message_bits.append(1)
        else:
            message_bits.append(0)
    return message_bits

def detect_motion_blocks_and_T(properties):
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
            flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
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

def encode_dtc_psyschovisual_lsb(orig_video_path, message_path, motion_blocks_file, generate_treshold=True):
    properties = vid_utils.video_to_rgb_frames(orig_video_path)
    message = bnr.string_to_binary_array(message_path)


        
    motion_blocks, T_values = detect_motion_blocks_and_T(properties)
    
    # Uložíme bloky pohybu do souboru
    save_motion_blocks(motion_blocks, motion_blocks_file)
    
    
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
        y_channel = yuv_image[:,:,0]
        block = y_channel[y:y+8, x:x+8]

        frame_message = message[message_index:message_index+bits_to_hide]
        if len(frame_message) < 3:
            frame_message = np.pad(frame_message, (0, 3 - len(frame_message)))

        dct_block = cv2.dct(np.float32(block))
        D = zigzag_create_D(dct_block)
        
        
        if generate_treshold:
            T = next(T_iterator, 20)
        else:
            T = 4
            
        f, s = setup_threshold_values(T, D)
        
        D = hiding_technique_abs(frame_message, f, s, D)
        
        dct_block = insert_zigzag_D(dct_block, D)
        
        modified_block = cv2.idct(dct_block)
        y_channel[y:y+8, x:x+8] = modified_block

        yuv_image[:,:,0] = y_channel

        modified_img = cv2.cvtColor(yuv_image, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(img_path, modified_img)

        message_index += bits_to_hide

    print(f"[INFO] Encoding completed. Encoded {message_index} bits.")
    vid_utils.reconstruct_video_from_rgb_frames(orig_video_path, properties)
    
    vid_utils.remove_dirs()

    return len(message)

def decode_dtc_psyschovisual_lsb(stego_video_path, len_b_msg, motion_blocks_file, write_file = False):
    properties = vid_utils.video_to_rgb_frames(stego_video_path)
    message = []
    
    motion_blocks = load_motion_blocks(motion_blocks_file)

    for frame, x, y in motion_blocks:
        if len(message) >= len_b_msg:
            break

        img_path = f"frames/frame_{frame}.png"
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = yuv_image[:,:,0]
        
        block = y_channel[y:y+8, x:x+8]
        
        dct_block = cv2.dct(np.float32(block))
        D = zigzag_create_D(dct_block)
        three_decoded_bits = extracting_technique_abs(D)
        
        for bit in three_decoded_bits:
            if len(message) < len_b_msg:
                message.append(bit)
            else:
                break

    print(f"Decoded message length: {len(message)} bits")
    decoded_message = bnr.binary_array_to_string(np.array(message))
    
    
    if write_file:
        output_path = "decoded_message.txt"
        string_utils.write_message_to_file(decoded_message,output_path)
        print(f"[INFO] saved decoded message as {output_path}")


    vid_utils.remove_dirs()
    

        
    return decoded_message



