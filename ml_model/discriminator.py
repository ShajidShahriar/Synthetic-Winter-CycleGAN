import torch
import torch.nn as nn

class Block(nn.Module):
    """
    A standard building block for the Discriminator.
    Structure: Conv2d -> InstanceNorm -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            # Kernel size 4, Stride 2 (halves the size), Padding 1
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) # Slope 0.2 lets some negative info through
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        
        # 1. Initial Layer (Raw Input)
        # We DO NOT use Normalization in the first layer of a discriminator.
        # Why? Because the first layer needs to see the raw color values (brightness/contrast).
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        in_channels = features[0]
        
        # 2. Hidden Layers (The Processing)
        # We loop through [64, 128, 256, 512] to build deeper layers
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
            
        # 3. Final Output Layer (The Judgment)
        # Compresses everything into 1 channel (Real/Fake score)
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        
        self.model = nn.Sequential(
            self.initial,
            *layers
        )

    def forward(self, x):
        # Returns raw scores (logits). 
        # We will use a Loss Function that handles the probability math later.
        return self.model(x)

# --- Sanity Check (Run this file to test) ---
if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256) # Fake image
    disc = Discriminator()
    preds = disc(x)
    print(f" Discriminator Output Shape: {preds.shape}")
    # Expected: [1, 1, 30, 30]