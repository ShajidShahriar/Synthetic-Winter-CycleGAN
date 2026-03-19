import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
   
 
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Reflection padding creates smoother edges
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels), 
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        # x is the original shape. 
        # self.block(x) is the style change.
        # Adding them together (x + ...) preserves the structure.
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator, self).__init__()

     
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh() 
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    
    print(" Initializing Generator")
    gen = Generator()
    
    print(" Running Test Passs")
    output = gen(x)
    
    print(f" Generator Output Shape: {output.shape}") 
    # Must be [1, 3, 256, 256]