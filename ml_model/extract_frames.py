import cv2
import os
from tqdm import tqdm

def extract_and_resize(video_path, output_folder, max_images=600, skip_frames=30):
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Missing {video_path}")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"created folder: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    
    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path}...")

    # Progress bar setup
    pbar = tqdm(total=max_images)

    while cap.isOpened() and saved_count < max_images:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Only save every Nth frame (skip_frames) to get variety
        if count % skip_frames == 0:
            # FORCE RESIZE to 256x256 (Critical for CycleGAN)
            small_frame = cv2.resize(frame, (256, 256))
            
            # Save the file
            filename = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(filename, small_frame)
            
            saved_count += 1
            pbar.update(1)

        count += 1

    cap.release()
    pbar.close()
    print(f"Saved {saved_count} images to {output_folder}")

# --- EXECUTION ---
print("Starting Extraction ")

# 1. Chop Summer video -> trainA
print("\nExtracting Summer Data...")
extract_and_resize("summer.mp4", "dataset/trainA", max_images=600, skip_frames=30)

# 2. Chop Winter video -> trainB
print("\n Extracting Winter Data...")
extract_and_resize("winter.mp4", "dataset/trainB", max_images=600, skip_frames=30)