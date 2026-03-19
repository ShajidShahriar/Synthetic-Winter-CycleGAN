import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# --- Import Your Custom Modules ---
# Make sure generator.py and discriminator.py are in the same folder
from dataset import DriveDataset
from generator import Generator
from discriminator import Discriminator

# --- Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset"
BATCH_SIZE = 1      # CycleGAN standard is 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 2     # Set to 0 if you get a "Broken Pipe" error on Windows
NUM_EPOCHS = 100
SAVE_MODEL = True
LOAD_MODEL = True # Set to True to resume training
CHECKPOINT_GEN_S = "gens.pth.tar"
CHECKPOINT_GEN_W = "genw.pth.tar"
CHECKPOINT_CRITIC_S = "critics.pth.tar"
CHECKPOINT_CRITIC_W = "criticw.pth.tar"

def save_checkpoint(model, optimizer, scaler, epoch, filename):
    print(f"=> Saving checkpoint to {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, scaler, lr):
    if not os.path.exists(checkpoint_file):
        print(f"=> No checkpoint found at {checkpoint_file}, starting from scratch")
        return 0
    print(f"=> Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        epoch = checkpoint.get("epoch", 0)
    else:
        # Fallback: assume the file is the state_dict itself
        model.load_state_dict(checkpoint)
        epoch = 0

    # If we don't do this then it will just use learning rate from checkpoint
    # which might be different from what we want
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    return epoch

def train_fn(disc_S, disc_W, gen_W, gen_S, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    loop = tqdm(loader, leave=True)

    for idx, (summer, winter) in enumerate(loop):
        summer = summer.to(DEVICE)
        winter = winter.to(DEVICE)

        # --- Train Discriminators (The Bullies) ---
        with torch.cuda.amp.autocast():
            # 1. Fake Winter
            fake_winter = gen_W(summer)
            D_W_real = disc_W(winter)
            D_W_fake = disc_W(fake_winter.detach())
            D_W_real_loss = mse(D_W_real, torch.ones_like(D_W_real))
            D_W_fake_loss = mse(D_W_fake, torch.zeros_like(D_W_fake))
            D_W_loss = D_W_real_loss + D_W_fake_loss

            # 2. Fake Summer
            fake_summer = gen_S(winter)
            D_S_real = disc_S(summer)
            D_S_fake = disc_S(fake_summer.detach())
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss

            # Average Loss
            D_loss = (D_W_loss + D_S_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # --- Train Generators (The Artists) ---
        with torch.cuda.amp.autocast():
            # Adversarial Loss (Did we fool the bully?)
            D_W_fake = disc_W(fake_winter)
            D_S_fake = disc_S(fake_summer)
            loss_G_W = mse(D_W_fake, torch.ones_like(D_W_fake))
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))

            # Cycle Loss (Original -> Fake -> Original)
            cycle_summer = gen_S(fake_winter)
            cycle_winter = gen_W(fake_summer)
            cycle_summer_loss = l1(summer, cycle_summer)
            cycle_winter_loss = l1(winter, cycle_winter)

            # Identity Loss (Winter -> Winter Gen -> Winter)
            # This stops the model from changing colors unnecessarily
            identity_summer = gen_S(summer)
            identity_winter = gen_W(winter)
            identity_summer_loss = l1(summer, identity_summer)
            identity_winter_loss = l1(winter, identity_winter)

            # Total Generator Loss
            G_loss = (
                loss_G_S
                + loss_G_W
                + cycle_summer_loss * LAMBDA_CYCLE
                + cycle_winter_loss * LAMBDA_CYCLE
                + identity_summer_loss * LAMBDA_IDENTITY
                + identity_winter_loss * LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Save images every 200 batches to verify progress
        if idx % 200 == 0:
            # Denormalize (x * 0.5 + 0.5) to view properly
            save_image(fake_winter * 0.5 + 0.5, f"saved_images/epoch{epoch}_batch{idx}_winter.png")
            save_image(fake_summer * 0.5 + 0.5, f"saved_images/epoch{epoch}_batch{idx}_summer.png")

        loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

def main():
    # --- Setup Transforms ---
    transforms = A.Compose(
        [
            A.Resize(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"},
    )

    print("📀 Loading Dataset...")
    dataset = DriveDataset(
        root_A="dataset/trainA", 
        root_B="dataset/trainB", 
        transform=transforms
    )
    
    # Check if we actually found images
    if len(dataset) == 0:
        print(" ERROR: Dataset is empty. Check your 'dataset/trainA' and 'dataset/trainB' folders!")
        sys.exit()

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    gen_W = Generator(input_nc=3, output_nc=3).to(DEVICE)
    gen_S = Generator(input_nc=3, output_nc=3).to(DEVICE)
    disc_W = Discriminator(in_channels=3).to(DEVICE)
    disc_S = Discriminator(in_channels=3).to(DEVICE)

    
    opt_disc = optim.Adam(
        list(disc_W.parameters()) + list(disc_S.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_W.parameters()) + list(gen_S.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    current_epoch = 0
    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_GEN_W, gen_W, opt_gen, g_scaler, LEARNING_RATE)
        load_checkpoint(CHECKPOINT_GEN_S, gen_S, opt_gen, g_scaler, LEARNING_RATE)
        load_checkpoint(CHECKPOINT_CRITIC_W, disc_W, opt_disc, d_scaler, LEARNING_RATE)
        current_epoch = load_checkpoint(CHECKPOINT_CRITIC_S, disc_S, opt_disc, d_scaler, LEARNING_RATE)

    # Create save folder
    if not os.path.exists("saved_images"):
        os.makedirs("saved_images")

    print(f" Training Started on {DEVICE}!")
    
    for epoch in range(current_epoch, NUM_EPOCHS):
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        train_fn(
            disc_S, disc_W, gen_W, gen_S, loader, 
            opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch
        )
        
        # Save checkpoints
        if SAVE_MODEL:
            save_checkpoint(gen_W, opt_gen, g_scaler, epoch, CHECKPOINT_GEN_W)
            save_checkpoint(gen_S, opt_gen, g_scaler, epoch, CHECKPOINT_GEN_S)
            save_checkpoint(disc_W, opt_disc, d_scaler, epoch, CHECKPOINT_CRITIC_W)
            save_checkpoint(disc_S, opt_disc, d_scaler, epoch, CHECKPOINT_CRITIC_S)

if __name__ == "__main__":
    main()