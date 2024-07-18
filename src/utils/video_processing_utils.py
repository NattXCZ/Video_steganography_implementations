import os
import shutil
from subprocess import run ,call ,STDOUT ,PIPE
from math import ceil

import cv2
import numpy as np


#TODO: mozna zkusit ukladat misto v .png tak v .avif


temporal_folder = "./tmp"
y_folder = "./tmp/Y"
u_folder = "./tmp/U"
v_folder = "./tmp/V"
rgb_folder = "./frames"

def decode_fourcc(cc):
    """
    Decodes a FourCC (four-character code) into its string representation.

    Examples:
        >>> decode_fourcc(875967080.0)  # Numerical input (0x34363248)
        'h264'
        >>> decode_fourcc(1668703592.0)  # Numerical input (0x64656370)
        'hevc'
        >>> decode_fourcc(808596553.0)  # Numerical input (0x30323449)
        'I420'
    """
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])
    

def delete_tmp(path="./tmp"):
    """Delete the temporary folder and its contents."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print("[INFO] tmp files are cleaned up")


def has_audio_track(filename, ffprobe_path=r".\src\utils\ffprobe.exe"):
    """Check if the given video file contains audio stream."""
    result = run([ffprobe_path, "-loglevel", "error", "-show_entries",
                             "stream=codec_type", "-of", "csv=p=0", filename],
                            stdout=PIPE,
                            stderr=PIPE,
                            text=True)
    
    # Check if 'audio' is in the result
    if 'audio' in result.stdout.split():
        return True
    return False


def get_vid_stream_bitrate(filename, ffprobe_path = r".\src\utils\ffprobe.exe"):
    """Get the bitrate of the first video stream in bits per second."""
    result = run([ffprobe_path, "-v", "quiet", "-select_streams", "v:0", 
                            "-show_entries", "stream=bit_rate", "-of",
                            "default=noprint_wrappers=1:nokey=1", filename],
        stdout=PIPE,
        stderr=STDOUT)
    return int(result.stdout)


def extract_audio_track(video_file, ffmpeg_path = r".\src\utils\ffmpeg.exe"):
    call([ffmpeg_path, "-i", video_file, "-aq", "0", "-map", "a", "tmp/audio.wav"]) 
    print("[INFO] audio extracted")


def video_to_rgb_frames(video_path):
    """Extracts frames from a video file and saves them as individual image files into "/frames" folder. Save as .png files."""
    if not os.path.exists("./frames"):
        os.makedirs("frames")
    temporal_folder="./frames"
    print("[INFO] frames directory is created")


    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error: Video file cannot be opened!")
        return


    video_properties = {
        "format": capture.get(cv2.CAP_PROP_FORMAT),
        "codec": capture.get(cv2.CAP_PROP_FOURCC),
        "container": capture.get(cv2.CAP_PROP_POS_AVI_RATIO),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    frame_count = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame_count += 1
        cv2.imwrite(os.path.join(temporal_folder, f"frame_{frame_count}.png"), frame)

    capture.release()


    print("[INFO] extraction finished")
    return video_properties


def video_to_yuv_frames(video_path):
    """Extracts frames from a video file and saves them as individual image files into "tmp" folder."""

    if not os.path.exists("./tmp"):
        os.makedirs("tmp")
    temporal_folder="./tmp"
    print("[INFO] tmp directory is created")


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
    
    png_compression_level = 0 
    frame_count = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame_count += 1


        frame_name = f"frame_{frame_count}.png"
      
        # Split frame into YUV components
        #YUV_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
        YUV_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        
        
        Y, U, V = cv2.split(YUV_frame)

        # Save Y, U, V components into respective folders
        y_path = os.path.join(temporal_folder, "Y")
        u_path = os.path.join(temporal_folder, "U")
        v_path = os.path.join(temporal_folder, "V")
        os.makedirs(y_path, exist_ok=True)
        os.makedirs(u_path, exist_ok=True)
        os.makedirs(v_path, exist_ok=True)

        cv2.imwrite(os.path.join(y_path, frame_name), Y,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
        cv2.imwrite(os.path.join(u_path, frame_name), U,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
        cv2.imwrite(os.path.join(v_path, frame_name), V,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])

    capture.release()

    print("[INFO] extraction finished")
    return video_properties


def reconstruct_video_from_rgb_frames(file_path, properties, ffmpeg_path = r".\src\utils\ffmpeg.exe"):
    """Reconstruct video from RGB frames with ffmpeg."""
    fps = properties["fps"]
    codec =  decode_fourcc(properties["codec"])
    #file_extension =  file_path.rsplit(".", 1)[1]
    file_extension = "avi"
    bitrate = get_vid_stream_bitrate(file_path)

    if has_audio_track(file_path):
        #extract audio stream from video
        extract_audio_track(file_path)
        
        #recreate video from frames (without audio)
        call([ffmpeg_path, "-r", str(fps), "-i", "frames/frame_%d.png" , "-vcodec", str(codec), "-b", str(bitrate),"-crf", "0","-pix_fmt", "yuv420p", f"tmp/video.{file_extension}", "-y"])

        #add audio to a recreated video
        call([ffmpeg_path, "-i", f"tmp/video.{file_extension}", "-i", "tmp/audio.wav","-q:v", "1", "-codec", "copy", f"video.{file_extension}", "-y"])
   
    else:
        #recreate video from frames (without audio)
        call([ffmpeg_path, "-r", str(fps), "-i", "frames/frame_%d.png","-q:v", "1", "-vcodec", str(codec), "-b", str(bitrate), "-pix_fmt", "yuv420p", f"video.{file_extension}", "-y"])

    print("[INFO] reconstruction is finished")
    

def merge_yuv_to_rgb(y_folder, u_folder, v_folder, output_folder):
    """Creates PNG frames by combining corresponding Y, U, and V frames. """
    os.makedirs(output_folder, exist_ok=True)

    y_files = sorted(os.listdir(y_folder))
    u_files = sorted(os.listdir(u_folder))
    v_files = sorted(os.listdir(v_folder))


    for y_file, u_file, v_file in zip(y_files, u_files, v_files):
        y_frame = cv2.imread(os.path.join(y_folder, y_file), cv2.IMREAD_GRAYSCALE)
        u_frame = cv2.imread(os.path.join(u_folder, u_file), cv2.IMREAD_GRAYSCALE)
        v_frame = cv2.imread(os.path.join(v_folder, v_file), cv2.IMREAD_GRAYSCALE)

        # merging Y, U, V frames into the one frame
        yuv_frame = cv2.merge([y_frame, u_frame, v_frame])

        # from YUV to RGB
        #rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)
        rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YCrCb2RGB)
        
        frame_name = os.path.splitext(y_file)[0] + ".png"
        cv2.imwrite(os.path.join(output_folder, frame_name), rgb_frame)
        
        
    print("[INFO] frames are merged")
        
        
def reconstruct_video_from_yuv_frames(file_path, properties, ffmpeg_path = r".\src\utils\ffmpeg.exe"):
    
    merge_yuv_to_rgb(y_folder, u_folder, v_folder, rgb_folder)
    
    reconstruct_video_from_rgb_frames(file_path, properties, ffmpeg_path)
    
    print("[INFO] video is reconstructed")
    
    
def rgb2yuv(image_name):
    frame_path = os.path.join(rgb_folder, image_name)
    rgb_image = cv2.imread(frame_path)
    
    if rgb_image is None:
        print(f"Error: Unable to load {frame_path}")
        return
    
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    #yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)

    # Split YUV channels
    Y = yuv_image[:, :, 0]
    U = yuv_image[:, :, 1]
    V = yuv_image[:, :, 2]


    y_path = os.path.join(y_folder, image_name)
    u_path = os.path.join(u_folder, image_name)
    v_path = os.path.join(v_folder, image_name)

    cv2.imwrite(y_path, Y)
    cv2.imwrite(u_path, U)
    cv2.imwrite(v_path, V)


def yuv2rgb(image_name):
    y_path = os.path.join(y_folder, image_name)
    u_path = os.path.join(u_folder, image_name)
    v_path = os.path.join(v_folder, image_name)

    Y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
    U = cv2.imread(u_path, cv2.IMREAD_GRAYSCALE)
    V = cv2.imread(v_path, cv2.IMREAD_GRAYSCALE)
    
    if Y is None or U is None or V is None:
        print(f"Error: Unable to load components for {image_name}")
        return

    yuv_image = np.stack((Y, U, V), axis=-1)


    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YCrCb2RGB)
    #rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

    rgb_path = os.path.join(rgb_folder, image_name)
    cv2.imwrite(rgb_path, rgb_image)
    
    
def create_dirs():
    """Create necessary directories."""
    if not os.path.exists(temporal_folder):
        os.makedirs("tmp")
        
    y_path = os.path.join(temporal_folder, "Y")
    u_path = os.path.join(temporal_folder, "U")
    v_path = os.path.join(temporal_folder, "V")
    
    os.makedirs(y_path, exist_ok=True)
    os.makedirs(u_path, exist_ok=True)
    os.makedirs(v_path, exist_ok=True)
    
    
def remove_dirs():
    """Remove temporary directories."""
    if os.path.exists(temporal_folder):
        shutil.rmtree(temporal_folder)
        print(f"[INFO] Removed {temporal_folder} and its subdirectories.")
    else:
        print(f"[INFO] {temporal_folder} does not exist.")
        
        
    if os.path.exists(rgb_folder):
        shutil.rmtree(rgb_folder)
        print(f"[INFO] Removed {rgb_folder}.")
    else:
        print(f"[INFO] {rgb_folder} does not exist.")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
################ I-frames, P-frames, B-frames ################
def create_dir(folder):
    if not os.path.exists(temporal_folder):
        os.makedirs("tmp")
        
    folder_path = os.path.join(temporal_folder, folder)
    
    os.makedirs(folder_path, exist_ok=True)
    
    
def move_all_files_to_destination_dir(original_dir, destination_dir):
    files = []
    for root, dirs, files in os.walk(original_dir):
        for name in files:
            os.rename(root + os.sep + name, destination_dir + os.sep + name)
    
def merge_files(directory):
    if not os.path.exists(rgb_folder):
        os.makedirs("frames")
    
    subdirs = [f.path for f in os.scandir(directory) if f.is_dir()] 
    for d in subdirs:
        #move_all_files_to_destination_dir(d, directory)
        move_all_files_to_destination_dir(d, r"./frames")
        # remove the subdirectories of d after we have moved all files under it
        for sub in os.scandir(d):
            if sub.is_dir():
                shutil.rmtree(sub)
        
        
def extract_frames_by_type(video_file, frame_type, ffmpeg_path = r".\src\utils\ffmpeg.exe"):
    create_dir(f"{frame_type}_frames")
    
    
    call([ffmpeg_path, "-i", video_file, "-vf", f"select=eq(pict_type\,{frame_type})", "-vsync", "vfr", "-frame_pts" , "true", f"tmp/{frame_type}_frames/frame_%d.png"]) 
    
    print(f"[INFO] {frame_type}-frames extracted")
    
def video_to_IBP_frames(video_path):
    
    extract_frames_by_type(video_path, "I")
    extract_frames_by_type(video_path, "B")
    extract_frames_by_type(video_path, "P")
    
    
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

def reconstruct_video_from_IBP_frames(file_path, properties, ffmpeg_path = r".\src\utils\ffmpeg.exe"):
    tmp_file = r"tmp"
    merge_files(tmp_file)
    reconstruct_video_from_rgb_frames(file_path, properties,ffmpeg_path)





def distribution_of_bits_between_frames(len_message, frame_count, n):
    codew_in_msg = ceil(len_message / n)
    codew_p_frame = codew_in_msg // frame_count
    tail = codew_in_msg - (codew_p_frame * frame_count) 
    
    return codew_p_frame, codew_p_frame + tail



