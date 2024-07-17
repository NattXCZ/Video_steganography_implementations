import numpy as np

from src.methods import dwt_method as bch_dwt
from src.utils import binary_utils as bnr

xor_key =  np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1])

def string_test():
    message = "Amet molestie ea diam et sadipscing vero ipsum in dolore et. Vero lorem sea vel nulla feugiat vero ea. Gubergren dolor labore tempor sanctus amet illum amet no erat. Dolor feugait ea kasd euismod clita est. Voluptua at aliquyam vero vulputate minim. Quod et sit takimata sit placerat sanctus takimata dolor eum magna dolor diam luptatum eos ipsum diam sit. Stet sed ipsum dolor kasd et at erat nonumy rebum erat ipsum nulla quod et dolor aliquyam ipsum. Dolores takimata takimata commodo dolore ipsum consetetur dolor sed magna iusto. Accusam laoreet duo congue clita et euismod ut et accusam. Dolores sadipscing odio ut sit dolor minim nulla sadipscing at takimata vel clita tincidunt. Aliquam dolores amet ad dolor ipsum invidunt dolor diam eros magna consetetur duo doming lorem volutpat ipsum aliquyam consetetur. Suscipit stet magna te amet erat ex at duo ea tempor est vero dolore est sea ut sanctus. Ipsum magna te stet sanctus ipsum takimata stet justo nonumy erat sea suscipit duo. Diam tation dolor diam luptatum sit ut takimata illum clita eirmod voluptua. Lobortis sanctus eum."
    #message = "Lorem ipsum dolor sit amet tempor consequat et sit dolore dolores vero ut et nisl eros ipsum no ipsum."
    num = 5
    
    vid_path = r"data_testing\input_video.mp4"
    codew_p_frame,codew_p_last_frame = bch_dwt.encode_bch_dwt(vid_path, message, xor_key,flag_delete_dirs=False, string_flag=True, bch_num=num)

    vid_sec = r"video.avi"
    outp_path = r"decoded.txt"
    
    #codew_p_frame = 1
    #codew_p_last_frame = 1 
    
    print(codew_p_frame, codew_p_last_frame)
    message_Dec = bch_dwt.decode_bch_dwt(vid_sec, codew_p_frame,codew_p_last_frame, xor_key,outp_path,flag_recostr_vid=False, string_flag=True, bch_num=num)

    print(message_Dec)
    
def file_test():

    #path = r"data_testing/carousel-horse.png"
    path = r"data_testing\angel_secret.txt"
    #path = r"data_testing\lorem_ipsum.txt"
    num = 5
    vid_path = r"data_testing\input_video.mp4"
    codew_p_frame,codew_p_last_frame = bch_dwt.encode_bch_dwt(vid_path, path, xor_key,flag_delete_dirs=False, bch_num=num)

    vid_sec = r"video.avi"
    outp_path = r"decoded.txt"
    #bch_dwt.decode_bch_dwt(vid_sec, codew_p_frame,codew_p_last_frame, xor_key,outp_path,flag_recostr_vid=False, bch_num=num)



if __name__ == "__main__":
    string_test()
    #file_test()
    

 
    