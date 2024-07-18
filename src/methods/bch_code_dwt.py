
    
        
import os

import numpy as np
import cv2
import pywt
from math import floor, ceil

from src.utils import video_processing_utils as vid_utils
from src.utils import binary_utils as bnr
from galois import BCH


#TODO: funguje pro text (v DWT ma nějake chyby) a nefunguje převod na video a zpět - stejný problém jako u LSB
#FIXME: chyba pri kratke zprave - kdyz je klic1 = 0
float_pairs = (
    (3.60, 0.65),  # Original values
    (3.61, 0.64),
    (3.59, 0.66),
    (3.62, 0.63),
    (3.58, 0.67),
    (3.63, 0.62),
    (3.57, 0.68),
    (3.64, 0.61)
)
def encode_bch_dwt(orig_video_path, message_path, xor_key,  string_flag = False, flag_delete_dirs = True, bch_num = 11):

    bch = BCH(15,bch_num)
    

    # Processing input message. Convert the secret information to a 1-D array, and after that change the position of  the entire message using chaotic algorithm 

    bin_arr = bnr.string_to_binary_array(message_path)



    # make message bits chaotic
    arr = process_binary_array(bin_arr)
    chaos_arr = fill_end_zeros(arr, bch_num)
    message_len = len(chaos_arr)


    #Extract frames in RGB. Every RGB frame is split into the YUV colour space.
    vid_properties = vid_utils.video_to_rgb_frames(orig_video_path)
    vid_utils.create_dirs()
    
    # extracting and saving Y,U,V components    
    for i in range(1, int(vid_properties["frames"]) + 1):
        image_name = f"frame_{i}.png"
        vid_utils.rgb2yuv(image_name)
    
    
    codew_p_frame, codew_p_last_frame =  vid_utils.distribution_of_bits_between_frames(message_len,vid_properties["frames"], bch_num)
    
    
    zero_key = False
    if codew_p_frame == 0:
        actual_max_codew = 1
        zero_key = True
    else:
        actual_max_codew = codew_p_frame
    
    
    
    embedded_codewords_per_frame = 0
        
    row = 0
    col = 0
    curr_frame = 1
    
    for i in range(0, len(chaos_arr), bch_num):
        msg_bits = chaos_arr[i:i+bch_num]
        
        codeword = bch.encode(msg_bits)
        
        codeword = codeword ^ xor_key 

        if embedded_codewords_per_frame == 0:

            y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
            u_component_path = f"./tmp/U/frame_{curr_frame}.png"
            v_component_path = f"./tmp/V/frame_{curr_frame}.png"

            y_frame = cv2.imread(y_component_path, cv2.IMREAD_GRAYSCALE)
            u_frame = cv2.imread(u_component_path, cv2.IMREAD_GRAYSCALE)
            v_frame = cv2.imread(v_component_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply 2D-DWT to U, V, and Y components
            coeffs_Y = pywt.dwt2(y_frame, 'haar')
            coeffs_U = pywt.dwt2(u_frame, 'haar')
            coeffs_V = pywt.dwt2(v_frame, 'haar')
            
            # Extract HH, HL, LH subbands
            LL_Y, (LH_Y, HL_Y, HH_Y) = coeffs_Y
            LL_U, (LH_U, HL_U, HH_U) = coeffs_U
            LL_V, (LH_V, HL_V, HH_V) = coeffs_V
            
            row = 0
            col = 0
            
            if i == 0:
                height , width = LL_Y.shape
        
        LH_Y_bin = format(floor(abs(LH_Y[row, col] + 0.0000001)), '#010b')
        HL_Y_bin = format(floor(abs(HL_Y[row, col] + 0.0000001 )), '#010b')
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
        
        col += 1

        if col >= width: 
            col = 0
            row += 1
            
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
            
            # Reconstruct frames using inverse DWT
            Y_new = pywt.idwt2((LL_Y, (LH_Y, HL_Y, HH_Y)), 'haar')
            U_new = pywt.idwt2((LL_U, (LH_U, HL_U, HH_U)), 'haar')
            V_new = pywt.idwt2((LL_V, (LH_V, HL_V, HH_V)), 'haar')
    
            cv2.imwrite(y_component_path, Y_new)
            cv2.imwrite(u_component_path, U_new)
            cv2.imwrite(v_component_path, V_new)
                
            curr_frame += 1
            if curr_frame == vid_properties["frames"]:
                actual_max_codew = codew_p_last_frame

        
    print(f"[INFO] encoded to frames")
    

    for i in range(1, int(vid_properties["frames"]) + 1):
        image_name = f"frame_{i}.png"
        vid_utils.yuv2rgb(image_name)
        
    vid_utils.reconstruct_video_from_rgb_frames(orig_video_path,vid_properties)
    

    if flag_delete_dirs:
        vid_utils.remove_dirs()
        
    
    print(f"[INFO] embedding finished")
    
    if zero_key:
        return 0, codew_p_last_frame

    return codew_p_frame, codew_p_last_frame



def decode_bch_dwt(stego_video_path, codew_p_frame, codew_p_last_frame, xor_key, output_path, string_flag = False, flag_recostr_vid = True, bch_num = 11):
    bch = BCH(15,bch_num)

    decoded_message = []
    codeword_chaos =  np.zeros(15, dtype = np.uint8)
    decoded_codeword = np.zeros(bch_num, dtype = np.uint8)
    

    zero_key = False
    if codew_p_frame == 0:
        actual_max_codew = 1
        zero_key = True
    else:
        actual_max_codew = codew_p_frame


    if flag_recostr_vid:
        vid_properties = vid_utils.video_to_rgb_frames(stego_video_path)
        
        vid_utils.create_dirs()
            
        # Etracting and saving Y,U,V components    
        for i in range(1, int(vid_properties["frames"]) + 1):
            image_name = f"frame_{i}.png"
            vid_utils.rgb2yuv(image_name)
    else:
        vid_properties = ret_properties(stego_video_path)
    
    
    for curr_frame in range(1, int(vid_properties["frames"]) + 1):
        embedded_codewords = 0

        #load new frame
        y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
        u_component_path = f"./tmp/U/frame_{curr_frame}.png"
        v_component_path = f"./tmp/V/frame_{curr_frame}.png"

        y_frame = cv2.imread(y_component_path, cv2.IMREAD_GRAYSCALE)
        u_frame = cv2.imread(u_component_path, cv2.IMREAD_GRAYSCALE)
        v_frame = cv2.imread(v_component_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply 2D-DWT to U, V, and Y components
        coeffs_Y = pywt.dwt2(y_frame, 'haar')
        coeffs_U = pywt.dwt2(u_frame, 'haar')
        coeffs_V = pywt.dwt2(v_frame, 'haar')
            
        # Extract HH, HL, LH subbands
        LL_Y, (LH_Y, HL_Y, HH_Y) = coeffs_Y
        LL_U, (LH_U, HL_U, HH_U) = coeffs_U
        LL_V, (LH_V, HL_V, HH_V) = coeffs_V
        
        if curr_frame == 1:
            height , width = LL_Y.shape
        
        if curr_frame == vid_properties["frames"]:
            actual_max_codew = codew_p_last_frame

        stop_loop = False

        for row in range(height):

            if stop_loop:
                break
            
            for col in range(width):

                if embedded_codewords >= actual_max_codew: 
                    stop_loop = True
                    break
                
                #* Obtain the encoded data from the YUV components and XOR with the random number using the same key that was used in the sender side.
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
                
                codeword = codeword_chaos ^ xor_key           #2x times XOR returns original codeword
                #codeword = codeword_chaos

                decoded_codeword = bch.decode(codeword)
                decoded_message.extend(decoded_codeword)
                
                embedded_codewords += 1
                
        #end of proceesing current frame
        if zero_key and codew_p_last_frame == curr_frame:
            break

    output_message = process_binary_array(np.array(decoded_message), False)



    diff = bch_num - (len(output_message) % bch_num)
    output_message = output_message[:-diff]

    message = bnr.binary_array_to_string(output_message)

    #vid_utils.remove_dirs()
    
    return message


########################### LOGISTIC MAPPING ###########################
def logistic_key(N, mu, x0):
    X = np.zeros(N)
    X[0] = mu * x0 * (1 - x0) 
    #X[0] =round(abs(mu * x0 * (1 - x0)), 4)
    for k in range(1, N):
        X[k] = mu * X[k-1] * (1 - X[k-1])
    return X
   
            
def create_B(logistic_seq):
    T = np.mean(logistic_seq)
    B = np.array([1 if x >= T else 0 for x in logistic_seq], dtype=np.uint8)
    return B


def process_part_component(binary_array, chaos_array, start_index, end_index, B):
    k = 0
    for i in range(start_index, end_index):
        C = binary_array[i]
        #B_val = 255 if B[k] == 1 else B[k]
        B_val = B[k]
        chaos_array[i] = C  ^ B_val
            
        k += 1
            
    return chaos_array


def process_binary_array(binary_array, encode = True):
    """Processes a 1D binary array by splitting it into 8 equal parts and adding process each part."""
    original_length = len(binary_array)
    
    # Check if the length of the array is divisible by 8
    if encode:
        if original_length % 8 != 0:
            # Calculate how many zeros need to be added
            padding_length = 8 - (original_length % 8)
            binary_array = np.append(binary_array, np.zeros(padding_length, dtype=np.uint8))
    else:
        len_8 = (original_length // 8) * 8
        if len_8 != original_length:
            diff = original_length - len_8
            binary_array = binary_array[:-diff]
            
        
    segment_length = len(binary_array) // 8
    
    chaos_array =  np.zeros_like(binary_array, dtype=np.uint8)
    
    i = 0
    for float_pair in float_pairs:
        start_index = i * segment_length
        end_index = start_index + segment_length
        
        generatedKey = logistic_key(segment_length, float_pair[0], float_pair[1])
        process_part_component(binary_array, chaos_array, start_index, end_index, create_B(generatedKey))
        
        i += 1

    return chaos_array


def check_11(st) :
    n = len(st) 
    
    odd_sum = 0
    even_sum = 0
    for i in range(0,n) :
        if (i % 2 == 0) :
            odd_sum = odd_sum + ((int)(st[i]))
        else:
            even_sum = even_sum + ((int)(st[i]))
     
    return ((odd_sum - even_sum) % 11 == 0)

def fill_end_zeros(array, num):
    length = len(array)
    if len(array) % num == 0:
        return array
    else:
        num_zeros = num - (length % num)
        adjusted_array = np.pad(array, (0, num_zeros), mode='constant', constant_values=0)

        return adjusted_array


def ret_properties(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error: Video file cannot be opened!")
        return

    # Get video properties before processing frames
    video_properties = {
        "format": capture.get(cv2.CAP_PROP_FORMAT),
        "codec": capture.get(cv2.CAP_PROP_FOURCC),
        "container": capture.get(cv2.CAP_PROP_POS_AVI_RATIO),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    
    return video_properties


#write message string
def write_message_to_file(message, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(message)

