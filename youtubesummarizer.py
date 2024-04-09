from pytube import Search
from pytube import YouTube
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import numpy as np
import imageio
import os
import easyocr


def download_top_video(search_string):
    s = Search(search_string)
    for video in s.results:
        yt = YouTube(video.watch_url)
        if yt.length < 240:  # 4 minutes = 240 seconds
            file_name = f"{yt.title}.mp4"
            # Check if the file already exists
            if not os.path.isfile(file_name):
                print(f"Downloading: {yt.title}")
                yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(filename=file_name)
                print("Download completed!")
            else:
                print(f"File already exists: {file_name}")
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
    
    print("Starting scene detection process")
    
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
    print("Starting key frames extraction process, this may take a few minuets")
    
    # Initialize video capture with the path to the video.
    cap = cv2.VideoCapture(video_path)
    
    # List to hold the key frames extracted from each scene.
    key_frames = []
    total_scenes = len(scenes)
    scene_index = 0
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
            
        progress_percentage = ((scene_index + 1) / total_scenes) * 100
        print(f"Progress: {progress_percentage:.2f}%", end='\r')
        scene_index += 1
        
        
    print("\nKey frame extraction complete.")
    # Release the video capture resources.
    cap.release()
    
    # Return the list of key frames.
    return key_frames

def find_key_frames_with_face_priority_optimized(video_path, scenes):
    """
    Optimized function to find key frames within each detected scene by analyzing frame-to-frame changes
    and prioritizing frames with faces.

    Parameters:
    - video_path (str): Path to the video file.
    - scenes (list of tuples): Each tuple contains the start and end frame numbers of a scene.

    Returns:
    - list of np.array: A list of key frames selected from the scenes, each as a NumPy array.
    """
    print("Starting key frames extraction process, this may take a few minuets")
    
    cap = cv2.VideoCapture(video_path)
    key_frames = []
    # Load the Haar cascade for face detection.
    face_cascade_path = r"haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    total_scenes = len(scenes)
    scene_index = 0

    for start, end in scenes:
        best_frame = None
        best_score = 0  # Initialize best score to 0

        for frame_num in range(start, end):
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if frame cannot be read

            # Convert frame to grayscale to reduce computation for face detection and diff calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Score frames based on the presence of faces, with a higher score for more faces
            face_score = len(faces) * 10000  # Give a high score for each face detected

            if frame_num == start:
                prev_frame = gray
                best_frame = frame
                best_score = face_score
            else:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                change_score = np.sum(diff)  # Use the sum of absolute differences as the change score
                
                # Combine scores, prioritizing face score
                total_score = face_score + change_score / 10000  # Adjust change score's influence

                if total_score > best_score:
                    best_frame = frame
                    best_score = total_score

            prev_frame = gray  # Update the previous frame for the next iteration

        if best_frame is not None:
            key_frames.append(best_frame)
            
        # Update and display progress bar
        progress_percentage = ((scene_index + 1) / total_scenes) * 100
        print(f"Progress: {progress_percentage:.2f}%", end='\r')
        scene_index += 1
        
    print("\nKey frame extraction complete.")
    cap.release()
    return key_frames

def detect_text_in_frame(frames):
    
    for frame in frames:
        # Initialize EasyOCR Reader
        reader = easyocr.Reader(['en'])  # Assuming English text; adjust the language as needed.
        
        # Ensure the frame is in the correct color format (EasyOCR expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect text in the frame
        results = reader.readtext(frame_rgb)
        
        if results:
            for (bbox, text, prob) in results:
                print(f"Detected text: {text} with confidence {prob:.2f}")
        # else:
        #     print("No text detected.")

def create_gif_from_frames(frames, output_path='summary.gif', fps=5, max_duration=10):
    print(f"{len(frames)} key frames were found.")
    # Adjust frames to fit the max_duration limit
    max_frames = fps * max_duration
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    if frames_rgb:
        imageio.mimsave(output_path, frames_rgb, fps=fps)
        print("GIF created successfully.")
    else:
        print("No frames to create a GIF. Exiting.")

# def find_key_frames_with_face_priority_optimized(video_path, scenes):
    # """
    # Optimized function to find key frames within each detected scene by analyzing frame-to-frame changes
    # and prioritizing frames with faces.

    # Parameters:
    # - video_path (str): Path to the video file.
    # - scenes (list of tuples): Each tuple contains the start and end frame numbers of a scene.

    # Returns:
    # - list of np.array: A list of key frames selected from the scenes, each as a NumPy array.
    # """
    # cap = cv2.VideoCapture(video_path)
    # key_frames = []
    # # Load the Haar cascade for face detection.
    # face_cascade_path = r"C:\Users\DanielSegal\anaconda3\pkgs\libopencv-4.9.0-qt6_py312hd35d245_612\Library\etc\haarcascades\haarcascade_frontalface_default.xml"
    # face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # for start, end in scenes:
    #     best_frame = None
    #     best_score = 0  # Initialize best score to 0

    #     for frame_num in range(start, end):
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    #         ret, frame = cap.read()
    #         if not ret:
    #             break  # Exit loop if frame cannot be read

    #         # Convert frame to grayscale to reduce computation for face detection and diff calculation
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         # Detect faces in the frame
    #         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
    #         # Score frames based on the presence of faces, with a higher score for more faces
    #         face_score = len(faces) * 1000  # Give a high score for each face detected

    #         if frame_num == start:
    #             prev_frame = gray
    #             best_frame = frame
    #             best_score = face_score
    #         else:
    #             # Calculate frame difference
    #             diff = cv2.absdiff(prev_frame, gray)
    #             change_score = np.sum(diff)  # Use the sum of absolute differences as the change score
                
    #             # Combine scores, prioritizing face score
    #             total_score = face_score + change_score / 10000  # Adjust change score's influence

    #             if total_score > best_score:
    #                 best_frame = frame
    #                 best_score = total_score

    #         prev_frame = gray  # Update the previous frame for the next iteration

    #     if best_frame is not None:
    #         key_frames.append(best_frame)

    # cap.release()
    # return key_frames

if __name__ == "__main__":
    # search_string = input("Enter the search string: ")
    # video_name = download_top_video(search_string)
    
    # print(video_name)

    # video_path = f'{video_name}.mp4'
    video_path = r'Dune Official Trailer.mp4'

    # Step 2: Detect scene changes.
    scenes = detect_scenes(video_path)

    # Step 3: Find key frames.
    # key_frames = find_key_frames_with_face_priority_optimized(video_path, scenes)
    key_frames = find_key_frames(video_path, scenes)
    
    # step 4: Detect text in each key frame
    detect_text_in_frame(key_frames)

    # Step 5: Create a GIF from the key frames.
    create_gif_from_frames(key_frames, 'summary4.gif', fps=5, max_duration=10)

