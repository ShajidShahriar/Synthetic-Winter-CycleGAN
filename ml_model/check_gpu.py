import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # The stress test
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    print(" Tensor math on GPU works! (RTX 3060 is awake)")
else:
    print("bro we are  on cpu. this will take 50 years ")