import cv2
import os

INPUT_DIR = "test_images_2"
RESULT_DIR = "test_results_2"
COMPARE_DIR = "comparisons_2"

# Create a folder to hold the side-by-side images
if not os.path.exists(COMPARE_DIR):
    os.makedirs(COMPARE_DIR)

print(f"Gluing images together and saving to '{COMPARE_DIR}'...")

for file in os.listdir(INPUT_DIR):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        # 1. Load Original Summer Image
        orig_path = os.path.join(INPUT_DIR, file)
        img1 = cv2.imread(orig_path)
        img1 = cv2.resize(img1, (256, 256))

        # 2. Load the Fake Winter Result
        res_path = os.path.join(RESULT_DIR, f"result_{file}")
        if not os.path.exists(res_path):
            continue 
            
        img2 = cv2.imread(res_path)

        # 3. Glue them together horizontally
        combined = cv2.hconcat([img1, img2]) 
        
        # 4. Save the result instead of trying to pop open a window
        save_path = os.path.join(COMPARE_DIR, f"compare_{file}")
        cv2.imwrite(save_path, combined)

print("Done! Open the 'comparisons_2' folder to see them.")