import numpy as np
import cv2
import pywt
from math import floor, ceil

import os
import shutil
from subprocess import run ,call ,STDOUT ,PIPE, check_output,CalledProcessError

import json
from src.utils import video_processing_utils as vid_utils
from src.utils import binary_utils as bnr
from src.utils.bcolors import bcolors 


import cv2
import numpy as np
import json


from skimage.metrics import structural_similarity as ssim

#zakomponovane zig zag ale ahzi duchy
#################
def zigzag_to_block(zigzag):
    block = np.zeros((8, 8), dtype=int)
    index = 0
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                block[i][j] = zigzag[index]
            else:
                block[j][i] = zigzag[index]
            index += 1
    return block

def block_to_zigzag(block):
    zigzag = []
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                zigzag.append(block[i][j])
            else:
                zigzag.append(block[j][i])
    return zigzag
###################
#write message string
def write_message_to_file(message, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(message)


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


def encode_dtc_psyschovisual(orig_video_path, message_path, string_flag=False, flag_delete_dirs=True):
    properties = vid_utils.video_to_IBP_frames(orig_video_path)
    
    if string_flag:
        message = bnr.string_to_binary_array(message_path)
    else:
        message = bnr.file_to_binary_1D_arr(message_path)
    
    print(f"{bcolors.OKCYAN}{message}{bcolors.ENDC}")
    

    motion_blocks, T_values = detect_motion_blocks_and_T(properties)
    
    T_iterator = iter(T_values)
    print(f"{bcolors.WARNING}{motion_blocks[0]}{bcolors.ENDC}")
        

    message_index = 0
    
    prtd = True
    testing = True
    msg_testing = []
    
    for frame, x, y in motion_blocks:
        if os.path.exists(f"tmp/P_frames/frame_{frame}.png"):
            img_path = f"tmp/P_frames/frame_{frame}.png"
        elif os.path.exists(f"tmp/B_frames/frame_{frame}.png"):
            img_path = f"tmp/B_frames/frame_{frame}.png"
        else:
            continue
        
        img = cv2.imread(img_path)
        yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = yuv_image[:, :, 0].astype(np.float32)
        
        block = Y[y:y+8, x:x+8]
        
        
        

        
        dct_block = cv2.dct(block)
        #!D = dct_block.flatten()[:6]  # Select first 6 DCT coefficients


        zigzag = block_to_zigzag(dct_block)
        D = np.array([zigzag[10], zigzag[11], zigzag[12], zigzag[13], zigzag[16], zigzag[15]])



        f, s = setup_threshold_values(get_next_T_value(T_iterator), D)
        
        if message_index < len(message):
            frame_message = message[message_index:message_index+3]
            if prtd:
                print(f"[{x}, {y}, frame = {frame}")
                print(f"{bcolors.WARNING}frame_message = {frame_message}{bcolors.ENDC}")
                print(f"{bcolors.WARNING}D orig{D}{bcolors.ENDC}")
                print("-------------")

            D_modified = hiding_technique_abs(frame_message, f, s, D)
            #D_modified += 4
            if prtd:
                print(f"{bcolors.WARNING}D {D_modified}{bcolors.ENDC}")
                print("-------------")
                #prtd = False
                #print(block)
                prtd = False
                
            message_index += 3
        else:
            break
        
        #!dct_block.flatten()[:6] = D_modified
        zigzag[10], zigzag[11], zigzag[12], zigzag[13], zigzag[16], zigzag[15] = D_modified #!D
        modified_dct_block = zigzag_to_block(zigzag)

        modified_block = cv2.idct(np.float32(modified_dct_block))
        #!modified_block = cv2.idct(dct_block)
        
        Y[y:y+8, x:x+8] = modified_block
        
        yuv_image[:, :, 0] = Y.astype(np.uint8)
        modified_img = cv2.cvtColor(yuv_image, cv2.COLOR_YCrCb2BGR)
        
        cv2.imwrite(img_path, modified_img)
        


        ####################################
        if testing:
            img = cv2.imread(img_path)
            yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            Y = yuv_image[:, :, 0].astype(np.float32)
            
            block = Y[y:y+8, x:x+8]
            
        
            dct_block = cv2.dct(block)
            
            
            #D = dct_block.flatten()[:6]  # Select first 6 DCT coefficients
            zigzag = block_to_zigzag(dct_block)
            D = np.array([zigzag[10], zigzag[11], zigzag[12], zigzag[13], zigzag[16], zigzag[15]])
            #print(f"{bcolors.WARNING}D after encode {D}{bcolors.ENDC}")

            
            
            
            message_bits = extracting_technique_abs(D)
            if len(frame_message) == len(message_bits):
                print(frame_message ,  message_bits)
                msg_testing.append(np.array(message_bits))
                if np.all(frame_message !=  message_bits):
                    print("-------------")
                    print(frame_message ==  message_bits)
                    print(f"D modified {D_modified}")
                    print(f"D rnwd {D}")
                    
                    print("-------------")
            
        ####################################    
    #vid_utils.reconstruct_video_from_IBP_frames(orig_video_path, properties)
    print(np.array(msg_testing).flatten())
    if flag_delete_dirs:
        vid_utils.remove_dirs()
    else:
        rgb_folder = "./frames"
        if os.path.exists(rgb_folder):
            print(f"[INFO] {rgb_folder} exists.")
        else:
            print(f"[INFO] {rgb_folder} does not exist.")

    return motion_blocks


def decode_dtc_psyschovisual(stego_video_path, message_len, output_path, properties, motion_blocks,  string_flag=False, flag_recostr_vid=False):
    if flag_recostr_vid:
        properties = vid_utils.video_to_IBP_frames(stego_video_path)
    
    output_message = []
    
    #!motion_blocks = detect_motion_blocks(properties)
    print(f"{bcolors.WARNING}{motion_blocks[0]}{bcolors.ENDC}")
    
    prtd = True
    
    for frame, x, y in motion_blocks:
        if os.path.exists(f"tmp/P_frames/frame_{frame}.png"):
            img_path = f"tmp/P_frames/frame_{frame}.png"
        elif os.path.exists(f"tmp/B_frames/frame_{frame}.png"):
            img_path = f"tmp/B_frames/frame_{frame}.png"
        elif os.path.exists(f"tmp/I_frames/frame_{frame}.png"):
            img_path = f"tmp/I_frames/frame_{frame}.png"
        else:
            if prtd:
                print(frame, x, y)
            continue
        
        img = cv2.imread(img_path)
        yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = yuv_image[:, :, 0].astype(np.float32)
        
        block = Y[y:y+8, x:x+8]
        
    
        dct_block = cv2.dct(block)
        
        
        #D = dct_block.flatten()[:6]  # Select first 6 DCT coefficients
        zigzag = block_to_zigzag(dct_block)
        D = np.array([zigzag[10], zigzag[11], zigzag[12], zigzag[13], zigzag[16], zigzag[15]])

        if prtd:
            print(f"[{x}, {y}] frame = {frame}")
            print("-----------------------")
            print(f"{bcolors.WARNING}D decoded{D}{bcolors.ENDC}")
            prtd = False
            #print(block)

        message_bits = extracting_technique_abs(D)
        output_message.extend(message_bits)
        
        if len(output_message) >= message_len:
            break
    
    output_message = output_message[:message_len]
    output_message = np.array(output_message)
    
    print(f"{bcolors.OKGREEN}{output_message}{bcolors.ENDC}")
    
    if string_flag:
        message = bnr.binary_array_to_string(output_message)
        if os.path.splitext(output_path)[1] == '.txt':
            write_message_to_file(message, output_path)
            print(f"[INFO] saved decoded message as {output_path}")
        else:
            print(f"[DECODED MESSAGE] {message}")
    else:
        bnr.binary_1D_arr_to_file(output_message, output_path)
        print(f"[INFO] saved decoded message as {output_path}")




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

    print("[INFO] extraction finished")
    return video_properties

#####################################
def reconstruct_video_with_IBP_frames(output_file, properties, ffmpeg_path=r".\src\utils\ffmpeg.exe"):
    fps = properties["fps"]
    codec = vid_utils.decode_fourcc(properties["codec"])
    width = properties["width"]
    height = properties["height"]
    frame_count = int(properties["frames"])
    
    # Vytvoříme dočasný soubor pro uložení informací o snímcích
    frame_info = []
    for frame in range(frame_count):
        if os.path.exists(f"tmp/I_frames/frame_{frame}.png"):
            frame_type = 'I'
        elif os.path.exists(f"tmp/P_frames/frame_{frame}.png"):
            frame_type = 'P'
        elif os.path.exists(f"tmp/B_frames/frame_{frame}.png"):
            frame_type = 'B'
        else:
            raise FileNotFoundError(f"Frame {frame} not found in any folder")
        
        frame_info.append({"frame": frame, "type": frame_type})
    
    with open("tmp/frame_info.json", "w") as f:
        json.dump(frame_info, f)
    
    # Vytvoříme seznam vstupních souborů pro FFmpeg
    input_files = []
    for frame in range(frame_count):
        if os.path.exists(f"tmp/I_frames/frame_{frame}.png"):
            input_files.append(f"tmp/I_frames/frame_{frame}.png")
        elif os.path.exists(f"tmp/P_frames/frame_{frame}.png"):
            input_files.append(f"tmp/P_frames/frame_{frame}.png")
        elif os.path.exists(f"tmp/B_frames/frame_{frame}.png"):
            input_files.append(f"tmp/B_frames/frame_{frame}.png")
    
    # Sestavíme FFmpeg příkaz
    ffmpeg_command = [
        ffmpeg_path,
        "-framerate", str(fps),
        "-i", "concat:" + "|".join(input_files),
        "-c:v", codec,
        "-g", "12",  # GOP size, můžete upravit podle potřeby
        "-bf", "2",  # Maximální počet B-snímků mezi I a P snímky
        "-flags", "+ildct+ilme",  # Povolí interlaced encoding, pokud je potřeba
        "-vf", f"scale={width}:{height}",
        "-pix_fmt", "yuv420p",
        "-metadata", f"frame_info_file=tmp/frame_info.json",
        output_file
    ]
    
    # Spustíme FFmpeg příkaz
    run(ffmpeg_command, check=True)
    
    print("[INFO] Video reconstruction with IBP frame information is finished")
    



def calculate_nc(img1, img2):
    img1_norm = cv2.normalize(img1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img2_norm = cv2.normalize(img2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    nc_value = np.mean(img1_norm * img2_norm)
    return nc_value

def determine_threshold(ssim_value, nc_value, alpha=0.5):
    T = alpha * ssim_value + (1 - alpha) * nc_value
    return T

def detect_motion_blocks_and_T(properties):
    motion_blocks = []
    T_values = []
    prev_frame = None

    for frame in range(int(properties['frames'])):
        if os.path.exists(f"tmp/P_frames/frame_{frame}.png"):
            img_path = f"tmp/P_frames/frame_{frame}.png"
        elif os.path.exists(f"tmp/B_frames/frame_{frame}.png"):
            img_path = f"tmp/B_frames/frame_{frame}.png"
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


def get_next_T_value(T_iterator):
    try:
        return next(T_iterator)
    except StopIteration:
        return None

    
def hiding_technique_abs(frame, f, s, D):
    S = 0
    for u in range(3):
        if S < len(frame):
            #jestli kodujeme 1 (na konci musi byt A < B)
            if frame[S] == 1:
                if abs(D[2*u]) < abs(D[2*u+1]):
                    D[2*u+1] += s[u]


                else:
                    C = D[2*u]
                    D[2*u] = D[2*u+1]
                    D[2*u+1] = C + f[u]

            #jestli kodujeme 0 (na konci musi byt A > B)
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



def main():
    video_file_path = r"data_testing\input_video.mp4"
    #message = "Hello how are you. I am fine what about you"
    message = "Lorem ipsum dolor sit amet nulla diam ipsum sea vero. Vero accusam magna tempor aliquyam. Dolore placerat magna labore amet voluptua dolor. Ipsum et no nulla erat. Ea sed lorem suscipit takimata in tincidunt sea dolore sea autem ut. Diam ut sadipscing ipsum esse duo suscipit consetetur kasd vero dolor lorem sit iusto stet amet. Sed eos erat nonumy ut sit ut amet aliquyam vero ipsum. Amet lorem voluptua aliquyam dolor dolores kasd ut sit. Nihil aliquyam gubergren cum lorem sea sit et ut wisi eum eos feugiat eos. Vero sadipscing nonumy takimata sea suscipit consetetur. Velit ea et et et ipsum est ut et erat. Id feugait erat lorem elitr consetetur consetetur vero justo eleifend sea lobortis eos et et vero. Eos zzril sadipscing tempor cum at possim molestie wisi. Gubergren voluptua aliquyam no vero magna diam. Dolores augue sadipscing eu delenit vel. Lorem kasd eum tempor ipsum ipsum dolor sit dolor erat sed sadipscing ut diam dolor magna erat. Tation molestie justo no stet labore tempor ad in et. Labore nostrud dolores sit lorem stet praesent sea consetetur. Iusto takimata voluptua dolore sit illum tincidunt consetetur eirmod lorem nonumy."
    #video_file_path = r"video.avi"
    motion_blocks = encode_dtc_psyschovisual(video_file_path, message, string_flag=True, flag_delete_dirs=False)
    
    stego_video_path = r"video.avi"
    output_path = r"./decoded_message.txt"
    properties = ret_properties(video_file_path)
    decode_dtc_psyschovisual(stego_video_path, len(message)*8, output_path,properties,motion_blocks, string_flag=True,flag_recostr_vid=True)


if __name__ == "__main__":
    main()

    

    
