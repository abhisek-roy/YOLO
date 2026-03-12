import torch

# Load checkpoint
ckpt = torch.load('/usr/local/aroy/Projects/YOLO/runs/train/v9-c/checkpoints/best-epoch=11-map=0.7025.ckpt', map_location='cpu')

# Check what's inside
print("Keys in checkpoint:", ckpt.keys())
print("\n")

# If 'state_dict' exists, check its keys
if 'state_dict' in ckpt:
    print("Model state_dict keys (first 10):")
    print(list(ckpt['state_dict'].keys())[:10])

