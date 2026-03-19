import torch
import os

checkpoints = ["gens.pth.tar", "genw.pth.tar", "critics.pth.tar", "criticw.pth.tar"]

for cp in checkpoints:
    if os.path.exists(cp):
        print(f"--- Checking {cp} ---")
        try:
            data = torch.load(cp, map_location="cpu")
            print(f"Keys: {list(data.keys())}")
            if "state_dict" not in data:
                print(f"!!! MISSING state_dict in {cp}")
        except Exception as e:
            print(f"!!! FAILED TO LOAD {cp}: {e}")
    else:
        print(f"--- {cp} does not exist ---")
