import torch
from pathlib import Path

MODEL_PATH = Path("src/models/unet_best.pt")

ckpt = torch.load(MODEL_PATH, map_location="cpu")

print(type(ckpt))

if isinstance(ckpt, dict):
    print("\nKeys:")
    for k in ckpt.keys():
        print(k)