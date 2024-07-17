import os

import cv2 as cv
import numpy as np
import random

from src.utils import binary_utils as bnr
from src.utils import video_processing_utils as vid_utils


#!tohle je verze bez row a col


#FIXME: Detekuje jen 1-bit chyby! Což dělá problém když se z videa extrahuje RGB snímek (ten se pak převádí na YUV).
# To dochází k drobným změnám v pixelech. (někdy se změní hodnota například jen z 162 na 161, z toho někdy z 142 na 138 a pod.)




######################################################################################################
# A Highly Secure Video Steganography using Hamming Code (7, 4)
#
# have three keys, shared between sender and receiver:
#   shift_key: Reposition pixels in Y, U, V, and the secret message into a random position, which makes the data chaotic
#
#   col_key and row_key:
#                           Used to pick the random rows and columns respectively in each chaotic Y, U and V component.
#


# Generator matrix
G = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1]
])


H_transposed = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 1],
    [1, 0, 1]
])

max_bits_per_frame = 22500    #maximum of bits saved in one frame (22500)
xor_key = np.array([1, 1, 1, 0, 0, 1, 1]) # 7-bit value 


def hamming_encode(orig_video_path, message_path, shift_key, col_key, row_key,xor_key, string_flag = False, shuffle_flag = True, flag_delete_dirs = True):
   
    #* Convert the video stream into individual frames. Separate each frame into its Y (luma), U (chrominance), and V (chrominance) components.
    vid_properties = vid_utils.video_to_rgb_frames(orig_video_path)
    vid_utils.create_dirs()
        
        
    # extracting and saving Y,U,V components    
    for i in range(1, int(vid_properties["frames"]) + 1):
        image_name = f"frame_{i}.png"
        vid_utils.rgb2yuv(image_name)
    

    # max amount of embedded pixels is 22% of all the pixels in frame (4 bits of message per one pixel)
    max_codew_p_frame = int(vid_properties["height"] * vid_properties["width"] * 0.22)
    
    
    #* Convert the message into a one-dimensional array, and then shift the entire message by a shift_key.
    #message = bnr.file_to_binary_1D_arr(message_path)
    #message = fill_end_zeros(np.roll(message, shift_key))
    #message = bnr.add_EOT_sequence(message)
    
    if string_flag:
        message = bnr.string_to_binary_array(message_path)
    else:
        message = bnr.file_to_binary_1D_arr(message_path)
    message = fill_end_zeros(np.roll(message, shift_key))
    
    
    
    # Count how much data will be stored in each frame
    codew_p_frame, codew_p_last_frame =  distribution_of_bits_between_frames(len(message),vid_properties["frames"])
    
    
    if codew_p_frame > max_codew_p_frame:
        print("[INFO] Message is too large to embed")
        return
    
    actual_max_codew = codew_p_frame
    
    
    #* Encode each 4-bit block of the message using a Hamming (7, 4) code.
    row = 0
    col = 0
    embedded_codewords_per_frame = 0
    curr_frame = 1
    
    for i in range(0, len(message), 4):
        four_bits = message[i:i+4]

        # Create codeword (Hamming code (7,4))
        codeword = (np.dot(four_bits, G) % 2)


        #* XOR the resulting 7-bit encoded data (4 message bits + 3 parity bits) with a random 7-bit value using a key.
        codeword = codeword ^ xor_key 


        #* Embed the resulting 7 bits into one pixel of the YUV components (3 bits in Y, 2 bits in U, and 2 bits in V).
        if embedded_codewords_per_frame == 0:
            y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
            u_component_path = f"./tmp/U/frame_{curr_frame}.png"
            v_component_path = f"./tmp/V/frame_{curr_frame}.png"

            y_frame = cv.imread(y_component_path, cv.IMREAD_GRAYSCALE)
            u_frame = cv.imread(u_component_path, cv.IMREAD_GRAYSCALE)
            v_frame = cv.imread(v_component_path, cv.IMREAD_GRAYSCALE)
            
            if shuffle_flag:
                #* Shift the position of all pixels in the Y, U, and V components by a specific key.
                y_frame_shuffled = seeded_shuffle_image(y_frame, shift_key)
                u_frame_shuffled = seeded_shuffle_image(u_frame, shift_key)
                v_frame_shuffled = seeded_shuffle_image(v_frame, shift_key)
            else:
                y_frame_shuffled = y_frame
                u_frame_shuffled = u_frame
                v_frame_shuffled = v_frame
            
            row = 0
            col = 0
            
        
        y_binary_value = format(y_frame_shuffled[row, col], '#010b')
        u_binary_value = format(u_frame_shuffled[row, col], '#010b')
        v_binary_value = format(v_frame_shuffled[row, col], '#010b')


        # Embed bits of codeword
        y_frame_shuffled[row, col] = int(y_binary_value[:-3] + ''.join(str(bit) for bit in codeword[:3]), 2)
        u_frame_shuffled[row, col] = int(u_binary_value[:-2] + ''.join(str(bit) for bit in codeword[3:5]), 2)
        v_frame_shuffled[row, col] = int(v_binary_value[:-2] + ''.join(str(bit) for bit in codeword[5:]), 2)
        
        
        embedded_codewords_per_frame += 1 
        
        # Update row and col after processing a pixel
        col += 1
        if col >= int(vid_properties["width"]):  # Reached end of current row
            col = 0
            row += 1
            
        if embedded_codewords_per_frame >= actual_max_codew:
            curr_frame += 1
            embedded_codewords_per_frame = 0
            
            if shuffle_flag:
                # Shift the positions of all pixels in the YUV components back to their original positions in the frame pixel grid.
                y_frame = seeded_unshuffle_image(y_frame_shuffled, shift_key)
                u_frame = seeded_unshuffle_image(u_frame_shuffled, shift_key)
                v_frame = seeded_unshuffle_image(v_frame_shuffled, shift_key)
            else:
                y_frame = y_frame_shuffled
                u_frame = u_frame_shuffled
                v_frame = v_frame_shuffled
            
            cv.imwrite(y_component_path, y_frame)
            cv.imwrite(u_component_path, u_frame)
            cv.imwrite(v_component_path, v_frame)
                
                
            if curr_frame == vid_properties["frames"]:
                actual_max_codew = codew_p_last_frame

        

    #FIXME:
    if embedded_codewords_per_frame > 0:
        cv.imwrite(y_component_path, y_frame)
        cv.imwrite(u_component_path, u_frame)
        cv.imwrite(v_component_path, v_frame)

    print(f"[INFO] encoded to frames")
    


    #* Reconstruct the video stream by combining the embedded frames
    for i in range(1, int(vid_properties["frames"]) + 1):
        image_name = f"frame_{i}.png"
        vid_utils.yuv2rgb(image_name)
        
    vid_utils.reconstruct_video_from_rgb_frames(orig_video_path,vid_properties)
    
    # for testing - delete dirs if flag_delete_dirs is true
    if flag_delete_dirs:
        vid_utils.remove_dirs()
        
    
    print(f"[INFO] embedding finished")
    return len(message)
    
    

def hamming_decode(stego_video_path, shift_key, col_key, row_key, message_len, output_path, xor_key,string_flag = False, shuffle_flag = True, flag_recostr_vid = True):
    #FIXME: list?
    decoded_message = []
    
    #zname delku tak:
    #decoded_message = np.zeros(message_len, dtype = np.uint8)
    
    
    codeword_chaos = np.array([0, 0, 0, 0, 0, 0, 0])
    #decoded_codeword = [] 
    decoded_codeword = np.array([0, 0, 0, 0])  

    #* Convert the video stream into frames. Separate each frame into Y, U and V components.
    if flag_recostr_vid:
        vid_properties = vid_utils.video_to_rgb_frames(stego_video_path)

        vid_utils.create_dirs()
            
        # Etracting and saving Y,U,V components    
        for i in range(1, int(vid_properties["frames"]) + 1):
            image_name = f"frame_{i}.png"
            vid_utils.rgb2yuv(image_name)
    else:
        stego_video_path = r"video.avi"
        vid_properties = ret_properties(stego_video_path)
        
        
            
            
    
    codew_p_frame, codew_p_last_frame =  distribution_of_bits_between_frames(message_len,vid_properties["frames"])
    actual_max_codew = codew_p_frame

    for curr_frame in range(1, int(vid_properties["frames"]) + 1):
        embedded_codewords = 0

        #load new frame
        y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
        u_component_path = f"./tmp/U/frame_{curr_frame}.png"
        v_component_path = f"./tmp/V/frame_{curr_frame}.png"

        y_frame = cv.imread(y_component_path, cv.IMREAD_GRAYSCALE)
        u_frame = cv.imread(u_component_path, cv.IMREAD_GRAYSCALE)
        v_frame = cv.imread(v_component_path, cv.IMREAD_GRAYSCALE)
        
        
        if shuffle_flag:
            #* Change the position of all pixel values in the three Y, U, and V components by the special key that was used in the embedding process.
            y_frame_shuffled = seeded_shuffle_image(y_frame, shift_key)
            u_frame_shuffled = seeded_shuffle_image(u_frame, shift_key)
            v_frame_shuffled = seeded_shuffle_image(v_frame, shift_key)
        else:
            y_frame_shuffled = y_frame
            u_frame_shuffled = u_frame
            v_frame_shuffled = v_frame
        
        
        
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
                
                #* Obtain the encoded data from the YUV components and XOR with the random number using the same key that was used in the sender side.
                y_binary_value = format(y_frame_shuffled[row, col], '#010b')
                u_binary_value = format(u_frame_shuffled[row, col], '#010b')
                v_binary_value = format(v_frame_shuffled[row, col], '#010b')
                
                
            

                codeword_chaos[0] = y_binary_value[-3]
                codeword_chaos[1] = y_binary_value[-2]
                codeword_chaos[2] = y_binary_value[-1]
    
                codeword_chaos[3] = u_binary_value[-2]
                codeword_chaos[4] = u_binary_value[-1]
    
                codeword_chaos[5] = v_binary_value[-2]
                codeword_chaos[6] = v_binary_value[-1]


                codeword = codeword_chaos ^ xor_key           #2x times XOR returns original codeword
                
                
                decoded_codeword = hamming_decode_codeword(codeword, H_transposed)
                decoded_message.extend(decoded_codeword)
                

                embedded_codewords += 1
                
                
                
        #end of proceesing current frame
        
    message_array = np.array(decoded_message)
    
    
    #* Reposition the whole message again into the original order.
    output_message = np.roll(message_array, - shift_key)
    #bnr.binary_1D_arr_to_file(np.roll(message_array, - shift_key), output_path) 
    
    if string_flag:
        message = bnr.binary_array_to_string(output_message)
        if os.path.splitext(output_path)[1] == '.txt':
            write_message_to_file(message,output_path)
            print(f"[INFO] saved decoded message as {output_path}")
        else:
            print(f"[DECODED MESSAGE] {message}")
    else:
        bnr.binary_1D_arr_to_file(output_message, output_path)  
        print(f"[INFO] saved decoded message as {output_path}")

    vid_utils.remove_dirs()
    
    if string_flag:
        return message

def hamming_decode_codeword(codeword, H_transposed):

    #Z = [(element % 2) for element in list(np.dot(codeword, H_transposed))]
    Z = (np.dot(codeword, H_transposed) % 2)
    R = codeword

    #find row representing a error
    index = -1
    for i, H_row in enumerate(H_transposed):
        if np.all(Z == H_row):            #Z == row:
         index = i      #indexing from 0
    
    #change bit on index (if there was an error)
    if index > -1:
        R[index] = 1 - R[index]

    #return first four bits
    return R[-4:]


##################
def fill_end_zeros(input_array):
    length = len(input_array)

    if length % 4 == 0:
        return input_array
    else:
        num_zeros = 4 - (length % 4)
        adjusted_array = np.pad(input_array, (0, num_zeros), mode='constant', constant_values=0)

        return adjusted_array
    

def distribution_of_bits_between_frames(len_message, frame_count):

    codew_in_msg = len_message // 4  # total number of words (divided by 4 bits per word)

    codew_p_frame, tail = divmod(codew_in_msg, frame_count)

    return codew_p_frame, codew_p_frame + tail


#shuffle
def seeded_shuffle_image(image, seed):
    rng = random.Random(seed)
    flattened = image.flatten()
    length = len(flattened)
    
    for i in range(length - 1, -1, -1):
        j = rng.randint(0, i)
        flattened[i], flattened[j] = flattened[j], flattened[i]
    
    return flattened.reshape(image.shape)


def seeded_unshuffle_image(image, seed):
    rng = random.Random(seed)
    flattened = image.flatten()
    length = len(flattened)
    
    indices = [rng.randint(0, i) for i in range(length - 1, -1, -1)]
    unshuffle_indices = list(range(length))
    
    for i, j in enumerate(indices[::-1]):
        unshuffle_indices[i], unshuffle_indices[j] = unshuffle_indices[j], unshuffle_indices[i]
    
    unshuffled = np.empty_like(flattened)
    for i, original_index in enumerate(unshuffle_indices):
        unshuffled[i] = flattened[original_index]
    
    return unshuffled.reshape(image.shape)


#write message string
def write_message_to_file(message, filename):

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(message)




######################

def ret_properties(video_path):
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error: Video file cannot be opened!")
        return

    # Get video properties before processing frames
    video_properties = {
        "format": capture.get(cv.CAP_PROP_FORMAT),
        "codec": capture.get(cv.CAP_PROP_FOURCC),
        "container": capture.get(cv.CAP_PROP_POS_AVI_RATIO),
        "fps": capture.get(cv.CAP_PROP_FPS),
        "frames": capture.get(cv.CAP_PROP_FRAME_COUNT),
        "width": int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    
    return video_properties