import os

import cv2 as cv
import numpy as np
import random

from src.utils import binary_utils as bnr
from src.utils import video_processing_utils as vid_utils


from subprocess import run ,call ,STDOUT ,PIPE




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



def hamming_encode(orig_video_path, message_path, shift_key, col_key, row_key, xor_key):
   
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

    message = bnr.string_to_binary_array(message_path)

    message = fill_end_zeros(np.roll(message, shift_key))
    
    
    
    # Count how much data will be stored in each frame
    codew_p_frame, codew_p_last_frame =  vid_utils.distribution_of_bits_between_frames(len(message),vid_properties["frames"], 4)

    
    
    
    zero_key = False
    if codew_p_frame == 0:
        actual_max_codew = 1
        zero_key = True
    else:
        #!proc je to dvakrat
        actual_max_codew = codew_p_frame
    
    
    
    
    if codew_p_frame > max_codew_p_frame:
        print("[INFO] Message is too large to embed")
        return

    
    #* Encode each 4-bit block of the message using a Hamming (7, 4) code.
    row = 0
    col = 0
    embedded_codewords_per_frame = 0
    curr_frame = 1
    
    for i in range(0, len(message), 4):
        four_bits = message[i:i+4]

        # Create codeword (Hamming code (7,4))
        codeword = hamming_encode_codeword(four_bits)


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
            
            
            row = 0
            col = 0
            
        
        y_binary_value = format(y_frame[row, col], '#010b')
        u_binary_value = format(u_frame[row, col], '#010b')
        v_binary_value = format(v_frame[row, col], '#010b')


        # Embed bits of codeword
        y_frame[row, col] = int(y_binary_value[:-3] + ''.join(str(bit) for bit in codeword[:3]), 2)
        u_frame[row, col] = int(u_binary_value[:-2] + ''.join(str(bit) for bit in codeword[3:5]), 2)
        v_frame[row, col] = int(v_binary_value[:-2] + ''.join(str(bit) for bit in codeword[5:]), 2)
        
        
        embedded_codewords_per_frame += 1 
        
        # Update row and col after processing a pixel
        col += 1
        if col >= int(vid_properties["width"]):  # Reached end of current row
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
    

    vid_utils.remove_dirs()
        
    
    print(f"[INFO] embedding finished")
    return len(message)
    
    

def hamming_decode(stego_video_path, shift_key, col_key, row_key, message_len, xor_key, write_file = False):
    decoded_message = []
    
    #zname delku tak:
    #decoded_message = np.zeros(message_len, dtype = np.uint8)
    
    
    codeword_chaos = np.array([0, 0, 0, 0, 0, 0, 0])
    decoded_codeword = np.array([0, 0, 0, 0])  

    #* Convert the video stream into frames. Separate each frame into Y, U and V components.

    vid_properties = vid_utils.video_to_rgb_frames(stego_video_path)

    vid_utils.create_dirs()
            
    # Etracting and saving Y,U,V components    
    for i in range(1, int(vid_properties["frames"]) + 1):
        image_name = f"frame_{i}.png"
        vid_utils.rgb2yuv(image_name)

        
            
            
    
    codew_p_frame, codew_p_last_frame =  vid_utils.distribution_of_bits_between_frames(message_len,vid_properties["frames"], 4)
    
    
    zero_key = False
    if codew_p_frame == 0:
        actual_max_codew = 1
        zero_key = True
    else:
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


                codeword = codeword_chaos ^ xor_key           #2x times XOR returns original codeword
                
                
                decoded_codeword = hamming_decode_codeword(codeword)
                decoded_message.extend(decoded_codeword)
                

                embedded_codewords += 1
                
            
            
        if zero_key and codew_p_last_frame == curr_frame:
            break
        
        
                
        #end of proceesing current frame
        
    message_array = np.array(decoded_message)
    
    
    #* Reposition the whole message again into the original order.
    output_message = np.roll(message_array, - shift_key)
    

    message = bnr.binary_array_to_string(output_message)
    if write_file:
        output_path = "decoded_message.txt"
        write_message_to_file(message,output_path)
        print(f"[INFO] saved decoded message as {output_path}")


    vid_utils.remove_dirs()
    
    return message


def hamming_encode_codeword(four_bits):
    return (np.dot(four_bits, G) % 2)
    
def hamming_decode_codeword(codeword):

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
    


#write message string
def write_message_to_file(message, filename):

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(message)




