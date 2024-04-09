from pytube import Search
from pytube import YouTube
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import numpy as np
import imageio


def download_top_video(search_string):
    # Perform the search
    s = Search(search_string)
    for video in s.results:
        # Get the video's duration in seconds
        yt = YouTube(video.watch_url)
        if yt.length < 600:  # 10 minutes = 600 seconds
            print(f"Downloading: {yt.title}")
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
            print("Download completed!")
            return yt.title
    else:
        print("No suitable video found.")
        
def detect_scenes(video_path):
    """
    Detect scenes in a video file based on changes in content.

    Parameters:
    - video_path (str): Path to the video file to analyze.

    Returns:
    - list of tuples: A list where each tuple represents a scene, containing the start and end frame numbers.
    """
    # Initialize the video manager with the path to the video.
    video_manager = VideoManager([video_path])
    
    # Create a scene manager instance to hold and use scene detectors.
    scene_manager = SceneManager()
    
    # Add a content detector to the scene manager. The content detector
    # looks for significant changes in the video content to identify scene boundaries.
    scene_manager.add_detector(ContentDetector())

    # Start processing the video file.
    video_manager.start()
    
    # Perform scene detection on the video file.
    scene_manager.detect_scenes(frame_source=video_manager)

    # Obtain the list of detected scenes, each represented as a start and end timecode.
    scenes = scene_manager.get_scene_list()
    
    # Release the video manager resources.
    video_manager.release()
    
    # Debugging: Print the number of detected scenes
    print(f"Detected {len(scenes)} scenes.")

    # Convert the scene list to a more usable format, specifically a list of tuples.
    # Each tuple contains the start and end frame numbers for a scene.
    return [(start.get_frames(), end.get_frames()) for start, end in scenes]

def find_key_frames(video_path, scenes):
    """
    Find key frames within each detected scene by analyzing the frame-to-frame changes.

    Parameters:
    - video_path (str): Path to the video file.
    - scenes (list of tuples): Each tuple contains the start and end frame numbers of a scene.

    Returns:
    - list of np.array: A list of key frames selected from the scenes, each as a NumPy array.
    """
    # Initialize video capture with the path to the video.
    cap = cv2.VideoCapture(video_path)
    
    # List to hold the key frames extracted from each scene.
    key_frames = []

    # Iterate through each scene to analyze its frames.
    for start, end in scenes:
        max_frame = None  # To hold the frame with the maximum change.
        max_diff = 0      # To hold the maximum frame difference value.
        prev_frame = None # To hold the previous frame for difference calculation.

        # Iterate through each frame in the current scene.
        for frame_num in range(start, end):
            # Set the video capture to the specific frame number.
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()  # Read the frame.
            if not ret:
                break  # Break the loop if the frame isn't read properly.

            # Convert the frame to grayscale to simplify the difference calculation.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate the difference from the previous frame, if it exists.
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                total_diff = np.sum(diff)  # Sum the differences to get a total difference value.

                # If the total difference is the highest so far, store the frame and its difference value.
                if total_diff > max_diff:
                    max_diff = total_diff
                    max_frame = frame

            prev_frame = gray  # Update the previous frame.

        # If a maximum difference frame was found, add it to the list of key frames.
        if max_frame is not None:
            key_frames.append(max_frame)

    # Release the video capture resources.
    cap.release()

    # Return the list of key frames.
    return key_frames

def create_gif_from_frames(frames, output_path='summary.gif', fps=1):
    # Convert frames to RGB and check if the list is empty
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    if not frames_rgb:
        print("No frames to create a GIF. Exiting.")
        return  # Exit the function if there are no frames to process

    imageio.mimsave(output_path, frames_rgb, fps=fps)



if __name__ == "__main__":
    search_string = input("Enter the search string: ")
    video_name = download_top_video(search_string)
    
    print(video_name)

    video_path = f'{video_name}.mp4'

    # Step 2: Detect scene changes.
    scenes = detect_scenes(video_path)

    # Step 3: Find key frames.
    key_frames = find_key_frames(video_path, scenes)

    # Step 4: Create a GIF from the key frames.
    create_gif_from_frames(key_frames, 'summary.gif', fps=1)

