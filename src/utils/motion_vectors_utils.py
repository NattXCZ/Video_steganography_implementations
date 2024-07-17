import numpy as np
import cv2
import os
from src.utils import video_processing_utils as vid_utils
from src.utils import binary_utils as bnr
import glob
######

# Uklada video s vyraznenymi MV na snimcich 
#funguje
def count_frames_in_directory(frame_type):
    file_path_pattern = f"tmp/{frame_type}_frames/frame_*.png"
    files = glob.glob(file_path_pattern)
    return len(files)

def visualize_motion_vectors(properties, motion_vectors):
    for frame in range(int(properties['frames'])):
        # Najdi odpovídající snímek
        if os.path.exists(f"tmp/P_frames/frame_{frame}.png"):
            img_path = f"tmp/P_frames/frame_{frame}.png"
        elif os.path.exists(f"tmp/B_frames/frame_{frame}.png"):
            img_path = f"tmp/B_frames/frame_{frame}.png"
        else:
            continue

        # Načti snímek v barevném formátu
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Vykresli vektory pohybu pro tento snímek
        for vector in motion_vectors:
            if vector[0] == frame:  # pokud vektor patří k tomuto snímku
                x, y = vector[1], vector[2]
                cv2.rectangle(img, (x, y), (x+8, y+8), (0, 255, 0), 1)  # zelený rámeček
                cv2.circle(img, (x+4, y+4), 2, (0, 0, 255), -1)  # červený bod uprostřed

        # Zobraz snímek
        cv2.imshow(f'Frame {frame}', img)
        cv2.waitKey(30)  # Počkej 30 ms nebo dokud uživatel nestiskne klávesu

    cv2.destroyAllWindows()




def save_motion_vectors(properties, motion_vectors):
    for frame in range(int(properties['frames'])):
        # Najdi odpovídající snímek
        if os.path.exists(f"tmp/P_frames/frame_{frame}.png"):
            img_path = f"tmp/P_frames/frame_{frame}.png"
        elif os.path.exists(f"tmp/B_frames/frame_{frame}.png"):
            img_path = f"tmp/B_frames/frame_{frame}.png"
        else:
            continue

        # Načti snímek v barevném formátu
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Vykresli vektory pohybu pro tento snímek
        for vector in motion_vectors:
            if vector[0] == frame:  # pokud vektor patří k tomuto snímku
                x, y = vector[1], vector[2]
                cv2.rectangle(img, (x, y), (x+8, y+8), (0, 255, 0), 1)  # zelený rámeček
                cv2.circle(img, (x+4, y+4), 2, (0, 0, 255), -1)  # červený bod uprostřed

        # Přepiš snímek na disku
        cv2.imwrite(img_path, img)
###################



def save_video_w_motion_vector_blocks(orig_video_path, message_path, string_flag=False, flag_delete_dirs=True):
    properties = vid_utils.video_to_IBP_frames(orig_video_path)

    if string_flag:
        message = bnr.string_to_binary_array(message_path)
    else:
        message = bnr.file_to_binary_1D_arr(message_path)
    
    motion_blocks = detect_motion_blocks(properties)
    
    #print(motion_blocks)
    save_motion_vectors(properties, motion_blocks)
    
    
    
    vid_utils.reconstruct_video_from_IBP_frames(orig_video_path, properties)
    
    if flag_delete_dirs:
        vid_utils.remove_dirs()
        
    print(message)



def detect_motion_blocks(properties):
    motion_blocks = []
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
            # Výpočet Dense Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Výpočet magnitudy pohybu
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Procházení bloků 8x8
            for i in range(0, current_frame.shape[0], 8):
                for j in range(0, current_frame.shape[1], 8):
                    block_magnitude = magnitude[i:i+8, j:j+8]
                    
                    # Výpočet průměrné magnitudy bloku
                    avg_magnitude = np.mean(block_magnitude)
                    
                    # Detekce pohybu, pokud průměrná magnituda je >= 7
                    #FIXME: puvodne 7
                    if avg_magnitude >= 7:    
                        motion_blocks.append((frame, j, i))

        prev_frame = current_frame

    return motion_blocks



def write_message_to_file(message, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(message)

def main():
    video_file_path = r"data_testing\input_video.mp4"
    message = "Hello how are you. I am fine what about you"
    
    save_video_w_motion_vector_blocks(video_file_path, message, string_flag=True)
    
    stego_video_path = r"video.avi"
    output_path = r"./decoded_message.txt"

if __name__ == "__main__":
    main()