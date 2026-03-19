import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class DriveDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        # Get list of all images
        self.images_A = os.listdir(root_A)
        self.images_B = os.listdir(root_B)
        
        # We need the dataset length to be the MAX of the two folders
        self.length_dataset = max(len(self.images_A), len(self.images_B))
        self.A_len = len(self.images_A)
        self.B_len = len(self.images_B)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        # Using % (modulo) allows us to reuse images if one folder is smaller
        img_file_A = self.images_A[index % self.A_len]
        img_file_B = self.images_B[index % self.B_len]

        path_A = os.path.join(self.root_A, img_file_A)
        path_B = os.path.join(self.root_B, img_file_B)

        # Open Images
        img_A = np.array(Image.open(path_A).convert("RGB"))
        img_B = np.array(Image.open(path_B).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=img_A, image0=img_B)
            img_A = augmentations["image"]
            img_B = augmentations["image0"]

        return img_A, img_B

# --- Verification ---
if __name__ == "__main__":
    # Test if we can find the folders
    # Note: We assume you run this from inside 'ml_model'
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Simple transform for testing
    test_transform = A.Compose(
        [A.Resize(256, 256), A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()],
        additional_targets={"image0": "image"},
    )
    
    # Point to your folders
    try:
        ds = DriveDataset("dataset/trainA", "dataset/trainB", transform=test_transform)
        print(f" Found {len(ds)} images.")
        imgA, imgB = ds[0]
        print(f" Image Shape: {imgA.shape}") 
    except Exception as e:
        print(f" Error: {e}")
        print(" check that 'dataset/trainA' exists inside 'ml_model' and has images")