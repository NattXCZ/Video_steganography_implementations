import os

import numpy as np
import cv2

#from src.methods import hamming_code as hmc
from src.methods import hamming_method as hmc
from src.utils import binary_utils as bnr


if __name__ == "__main__":

    #typy toho co muzeme vkladat
    message_text_file = r"data_testing/angel_secret.txt"
    
    message_text_string = "Lorem ipsum dolor sit amet congue. Ut consectetuer lorem est labore dolor. Ut quis assum erat at dolores zzril justo aliquyam aliquyam nibh eirmod labore est voluptua te sit illum. Feugiat eirmod vel no invidunt facilisis. Sed amet et vero consetetur eos sea tempor luptatum voluptua. Dolore labore sanctus et justo erat sed rebum accusam lorem illum elitr accusam takimata nonumy. Sanctus erat consequat nobis nisl amet eleifend et rebum justo lorem clita nisl. Sit velit eleifend sadipscing rebum ex. Veniam et lorem aliquyam sanctus exerci consetetur. No clita amet eirmod vulputate duo vero amet sadipscing. Magna nobis stet nonumy et aliquam. Gubergren accusam ipsum facilisi sed."
    message_png_file = r"data_testing/carousel-horse.png"
    
    #cesty k video souborum (vstupni video ve ktere vkladame a stego video)        
    video_path = r"data_testing/input_video.mp4"
    stego_video_path = r"video.avi"
    
    #klice
    key_1 = 10
    key_2 = 20
    key_3 = 2

    #kam zapsat tajnou zpravu
    output_path = r"./decoded_message.txt"
    output_path_png = r"./carousel-horse_decoded.png"
    

    #promenna flag ridi jakou cast testujeme
    flag = 3
    
    #promenna shuffle ridi jestli chceme vyuzivat michani pixelu snimku pred vlozenim - s nim vypocet někdy trva dlouho proto je ted na False
    #ale lze prepnout na True
    shuffle = False
    
    #jestli je True - extrahuje snimky z videa, vytvori z RGB snimku ty YUV snimky
    #jestli False - bude provadet vsechny operace na jiz extrahovanych YUV snimcich ktere zustaly 
    # po metode "hamming_encode" (v te se ale musí flag_delete_dirs nastavit na False - to ale resi tato promenna taky)
    recontructed_vid = False
    
    
    
    xor_key = np.array([1, 1, 1, 0, 0, 1, 1]) # 7-bit value
    
    #SOUBOR
    if flag == 1:
        #zakodovani souboru do videa (tady zakoduje obrazek png)
        hmc.hamming_encode(video_path,message_png_file, key_1, key_2, key_3,xor_key, shuffle_flag = shuffle, flag_delete_dirs = recontructed_vid)
    elif flag == 2:
        #dekodovani souboru z videa (zde png)
        
        message_len_file = len(bnr.file_to_binary_1D_arr(message_png_file))
        hmc.hamming_decode(stego_video_path, key_1, key_2, key_3, message_len_file, output_path_png, xor_key, shuffle_flag = shuffle, flag_recostr_vid = recontructed_vid)



    #STRING
    elif flag == 3:
        #zakodovani retezce do videa
        hmc.hamming_encode(video_path,message_text_string, key_1, key_2, key_3, xor_key, string_flag = True, shuffle_flag = shuffle,flag_delete_dirs = recontructed_vid)
        
        
    elif flag == 4:
        #dekodovani retezce z videa

        message_len_string = len(bnr.string_to_binary_array(message_text_string))
        hmc.hamming_decode(stego_video_path, key_1, key_2, key_3, message_len_string, output_path,xor_key, string_flag = True, shuffle_flag = shuffle, flag_recostr_vid = recontructed_vid)


