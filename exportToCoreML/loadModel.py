import sys
import torch
sys.path.append('/usr/local/aroy/Projects/YOLO')

from yolo.model.yolo import create_model
from omegaconf import OmegaConf

# Load checkpoint
ckpt = torch.load('/usr/local/aroy/Projects/YOLO/runs/train/v9-c/checkpoints/best-epoch=11-map=0.7025.ckpt', map_location='cpu')

# Load model config
model_cfg = OmegaConf.load('/usr/local/aroy/Projects/YOLO/yolo/config/model/v9-c.yaml')

# Create model with correct class number (11 for DocLayNet)
model = create_model(model_cfg, class_num=11, weight_path=None)

# Extract EMA weights from checkpoint
state_dict = {}
for k, v in ckpt['state_dict'].items():
    # Use EMA weights (best for inference)
    if k.startswith('ema.model.'):
        new_key = k.replace('ema.model.', 'model.')
        state_dict[new_key] = v

# Load weights
model.load_state_dict(state_dict, strict=False)
model.eval()

print("✅ Model loaded successfully!")
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
# Test with dummy input
dummy_input = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    output = model(dummy_input)
    print("\n📊 Output structure:")
    for key, value in output.items():
        print(f"\n{key}:")
        for i, item in enumerate(value):
            print(f"  Scale {i}: type={type(item)}")
            if isinstance(item, tuple):
                print(f"    Tuple length: {len(item)}")
                for j, tensor in enumerate(item):
                    if torch.is_tensor(tensor):
                        print(f"      [{j}] shape: {tensor.shape}")
                    else:
                        print(f"      [{j}] type: {type(tensor)}")
            elif torch.is_tensor(item):
                print(f"    Tensor shape: {item.shape}")
