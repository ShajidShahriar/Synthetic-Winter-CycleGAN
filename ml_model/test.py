import torch
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os
from generator import Generator

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE = "genw.pth.tar"  # The file saved by train.py
INPUT_FOLDER = "test_images_2"
OUTPUT_FOLDER = "test_results_2"

def load_checkpoint(checkpoint_file, model):
    print(f"Loading weights from {checkpoint_file}...")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    # Handle dictionary vs raw state_dict loading
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
         model.load_state_dict(checkpoint["state_dict"])
    else:
         model.load_state_dict(checkpoint)

def main():
    # 1. Setup
    gen_W = Generator(input_nc=3, output_nc=3).to(DEVICE)
    
    # 2. Load the Trained Weights
    if not os.path.exists(CHECKPOINT_FILE):
        print(f" Error: Cannot find {CHECKPOINT_FILE}. Did you finish training?")
        return

    load_checkpoint(CHECKPOINT_FILE, gen_W)
    gen_W.eval() # Set to "Evaluation Mode" (Turns off Dropout/Batchnorm updates)

    # 3. Create Output Folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 4. Processing Loop
    transform = A.Compose(
        [
            A.Resize(width=256, height=256),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    print(f" converting images from '{INPUT_FOLDER}'...")
    
    files = os.listdir(INPUT_FOLDER)
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load Image
            img_path = os.path.join(INPUT_FOLDER, file)
            original_img = np.array(Image.open(img_path).convert("RGB"))
            
            # Prepare for AI
            transformed = transform(image=original_img)["image"].unsqueeze(0).to(DEVICE)
            
            # Generate Winter
            with torch.no_grad(): # Don't calculate gradients (saves RAM)
                fake_winter = gen_W(transformed)
            
            # Save Result
            # We save them side-by-side for easy comparison
            save_name = f"result_{file}"
            save_path = os.path.join(OUTPUT_FOLDER, save_name)
            
            # Denormalize (x * 0.5 + 0.5)
            save_image(fake_winter * 0.5 + 0.5, save_path)
            print(f" Saved: {save_name}")

if __name__ == "__main__":
    main()