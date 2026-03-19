import cv2
import os
from tqdm import tqdm

def extract_and_resize(video_path, output_folder, max_images=20, skip_frames=60, start_sec=0, end_sec=None):
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Missing {video_path}")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f" Created folder: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    
    # THE MAGIC TRICK: Jump straight to the start time (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    
    count = 0
    saved_count = 0
    print(f" Processing {video_path} starting at {start_sec} seconds...")

    # Progress bar setup
    pbar = tqdm(total=max_images)

    while cap.isOpened() and saved_count < max_images:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Check where we are in the video
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Stop if we hit the end time limit
        if end_sec is not None and current_sec > end_sec:
            print(f"\n Reached end time of {end_sec}s! Stopping.")
            break

        # Only save every Nth frame
        if count % skip_frames == 0:
            small_frame = cv2.resize(frame, (512, 512))
            filename = os.path.join(output_folder, f"test_frame_{saved_count}.jpg")
            cv2.imwrite(filename, small_frame)
            
            saved_count += 1
            pbar.update(1)

        count += 1

    cap.release()
    pbar.close()
    print(f"Saved {saved_count} images to {output_folder}")

# --- EXECUTION ---
print("Starting Test Data Extraction")

# Modify these times to wherever the good part of your YouTube video is!
# For example: Start at 2 mins 15 secs (135s) and end at 5 mins (300s)
START_TIME_IN_SECONDS = 600  
END_TIME_IN_SECONDS = 800   

extract_and_resize(
    video_path="test2.mp4", 
    output_folder="./dataset/trainA", 
    max_images=20, 
    skip_frames=60, 
    start_sec=START_TIME_IN_SECONDS, 
    end_sec=END_TIME_IN_SECONDS
)