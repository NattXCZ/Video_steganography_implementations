import os
import shutil
from subprocess import run, call, PIPE
from math import ceil
import json
import cv2

# Global variables
TEMPORAL_FOLDER = "./tmp"
Y_FOLDER = "./tmp/Y"
U_FOLDER = "./tmp/U"
V_FOLDER = "./tmp/V"
RGB_FOLDER = "./frames"


def has_audio_track(filename, ffprobe_path=r".\src\utils\ffprobe.exe"):
    """Check if the given video file contains an audio stream."""
    result = run([ffprobe_path, "-loglevel", "error", "-show_entries",
                  "stream=codec_type", "-of", "csv=p=0", filename],
                 stdout=PIPE,
                 stderr=PIPE,
                 text=True)
    
    return 'audio' in result.stdout.split()


def extract_audio_track(video_file, ffmpeg_path=r".\src\utils\ffmpeg.exe"):
    """Extract the audio track from a video file."""
    call([ffmpeg_path, "-i", video_file, "-aq", "0", "-map", "a", "tmp/audio.wav"])
    print("[INFO] audio extracted")


def video_to_rgb_frames(video_path):
    """Extract frames from a video file and save them as individual PNG files."""
    if not os.path.exists("./frames"):
        os.makedirs("frames")
    temporal_folder = "./frames"
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


def rgb2yuv(image_name):
    """Convert an RGB image to YUV color space and save the components."""
    frame_path = os.path.join(RGB_FOLDER, image_name)
    rgb_image = cv2.imread(frame_path)
    
    if rgb_image is None:
        print(f"Error: Unable to load {frame_path}")
        return
    
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)

    Y, U, V = cv2.split(yuv_image)

    cv2.imwrite(os.path.join(Y_FOLDER, image_name), Y)
    cv2.imwrite(os.path.join(U_FOLDER, image_name), U)
    cv2.imwrite(os.path.join(V_FOLDER, image_name), V)


def yuv2rgb(image_name):
    """Convert YUV components back to an RGB image."""
    y_path = os.path.join(Y_FOLDER, image_name)
    u_path = os.path.join(U_FOLDER, image_name)
    v_path = os.path.join(V_FOLDER, image_name)

    Y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
    U = cv2.imread(u_path, cv2.IMREAD_GRAYSCALE)
    V = cv2.imread(v_path, cv2.IMREAD_GRAYSCALE)
    
    if Y is None or U is None or V is None:
        print(f"Error: Unable to load components for {image_name}")
        return

    yuv_image = cv2.merge([Y, U, V])

    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YCrCb2RGB)

    rgb_path = os.path.join(RGB_FOLDER, image_name)
    cv2.imwrite(rgb_path, rgb_image)


def create_dirs():
    """Create necessary directories for temporary files."""
    os.makedirs(TEMPORAL_FOLDER, exist_ok=True)
    
    os.makedirs(Y_FOLDER, exist_ok=True)
    os.makedirs(U_FOLDER, exist_ok=True)
    os.makedirs(V_FOLDER, exist_ok=True)


def remove_dirs():
    """Remove temporary directories."""
    if os.path.exists(TEMPORAL_FOLDER):
        shutil.rmtree(TEMPORAL_FOLDER)
        print(f"[INFO] Removed {TEMPORAL_FOLDER} and its subdirectories.")
    else:
        print(f"[INFO] {TEMPORAL_FOLDER} does not exist.")
        
    if os.path.exists(RGB_FOLDER):
        shutil.rmtree(RGB_FOLDER)
        print(f"[INFO] Removed {RGB_FOLDER}.")
    else:
        print(f"[INFO] {RGB_FOLDER} does not exist.")


def distribution_of_bits_between_frames(len_message, frame_count, n):
    """Calculate the distribution of bits between frames."""
    codew_in_msg = ceil(len_message / n)
    codew_p_frame = codew_in_msg // frame_count
    tail = codew_in_msg - (codew_p_frame * frame_count) 
    
    return codew_p_frame, codew_p_frame + tail


def reconstruct_video_from_rgb_frames(file_path, properties, ffmpeg_path=r".\src\utils\ffmpeg.exe"):
    """Reconstruct video from RGB frames using ffmpeg."""
    fps = properties["fps"]
    file_extension = "avi"

    if has_audio_track(file_path):
        # Extract audio stream from video
        extract_audio_track(file_path)
        
        # Recreate video from frames (without audio)
        call([ffmpeg_path, "-r", str(fps), "-i", "frames/frame_%d.png", 
              "-vcodec", "ffv1", 
              "-level", "3", 
              "-coder", "1", 
              "-context", "1", 
              "-g", "1", 
              "-pix_fmt", "bgr0",
              "-color_range", "pc",
              "tmp/video.avi", "-y"])
        
        # Add audio to a recreated video
        call([ffmpeg_path, "-i", f"tmp/video.{file_extension}", "-i", "tmp/audio.wav",
              "-q:v", "1", "-codec", "copy", f"video.{file_extension}", "-y"])
   
    else:
        # Recreate video from frames (without audio)
        call([ffmpeg_path, "-r", str(fps), "-i", "frames/frame_%d.png", 
              "-vcodec", "ffv1", 
              "-level", "3", 
              "-coder", "1", 
              "-context", "1", 
              "-g", "1", 
              "-pix_fmt", "bgr0",
              "-color_range", "pc",
              "video.avi", "-y"])
        
    print("[INFO] reconstruction is finished")
    
    
def get_I_frame_numbers(video_file, ffprobe_path="ffprobe"):
    """Get the numbers of I-frames in a video."""
    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "packet=pts,flags",
        "-of", "json",
        video_file
    ]
    
    result = run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    i_frames = [index for index, packet in enumerate(data['packets']) if 'K' in packet['flags']]
    
    return i_frames