import sys
import torch
import torch.nn as nn

sys.path.append('/usr/local/aroy/Projects/YOLO')

from yolo.model.yolo import create_model
from omegaconf import OmegaConf

# ========== LOAD MODEL ==========

print("📥 Loading checkpoint...")
ckpt = torch.load(
    '/usr/local/aroy/Projects/YOLO/runs/train/v9-c/checkpoints/best-epoch=11-map=0.7025.ckpt',
    map_location='cpu'
)

print("📐 Loading model config...")
model_cfg = OmegaConf.load('/usr/local/aroy/Projects/YOLO/yolo/config/model/v9-c.yaml')

print("🏗️  Creating model...")
model = create_model(model_cfg, class_num=11, weight_path=None)

print("⚖️  Loading weights...")
state_dict = {}
for k, v in ckpt['state_dict'].items():
    if k.startswith('ema.model.'):
        new_key = k.replace('ema.model.', 'model.')
        state_dict[new_key] = v

model.load_state_dict(state_dict, strict=False)
model.eval()

print(f"✅ Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ========== CREATE WRAPPER (UPDATED) ==========

class YOLOv9Wrapper(nn.Module):
    """Wrapper with flattened outputs for CoreML export"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Get only Main head output
        output = self.model(x)
        main_output = output['Main']
        
        # Flatten the nested structure
        # main_output is a list of 3 scales, each containing (class_pred, box_dist, box_coord)
        
        # Scale 0 (128x128)
        class_pred_0, box_dist_0, box_coord_0 = main_output[0]
        
        # Scale 1 (64x64)
        class_pred_1, box_dist_1, box_coord_1 = main_output[1]
        
        # Scale 2 (32x32)
        class_pred_2, box_dist_2, box_coord_2 = main_output[2]
        
        # Return as individual tensors (9 outputs total)
        return (
            class_pred_0, box_dist_0, box_coord_0,
            class_pred_1, box_dist_1, box_coord_1,
            class_pred_2, box_dist_2, box_coord_2
        )

# Wrap the model
wrapped_model = YOLOv9Wrapper(model)
wrapped_model.eval()

print("\n🎁 Model wrapped successfully!")

# ========== TEST WRAPPER ==========

print("\n🧪 Testing wrapped model...")
dummy_input = torch.randn(1, 3, 1024, 1024)

with torch.no_grad():
    test_output = wrapped_model(dummy_input)
    print("\n📦 Wrapped model output structure:")
    for i, scale in enumerate(test_output):
        print(f"  Scale {i}: {len(scale)} tensors")
        for j, tensor in enumerate(scale):
            print(f"    [{j}] shape: {tensor.shape}")

print("\n✅ Wrapper test complete!")

# ========== TRACE MODEL ==========

print("\n🔍 Tracing model with TorchScript...")

example_input = torch.randn(1, 3, 1024, 1024)

try:
    traced_model = torch.jit.trace(wrapped_model, example_input)
    print("✅ Model traced successfully!")
    
    # Test traced model
    with torch.no_grad():
        traced_output = traced_model(example_input)
        print("\n📊 Traced model output:")
        print(f"  Type: {type(traced_output)}")
        print(f"  Number of tensors: {len(traced_output)}")
        for i, tensor in enumerate(traced_output):
            print(f"  Output {i}: shape {tensor.shape}")
    
    # Save traced model
    traced_model_path = "/usr/local/aroy/Projects/YOLO/runs/train/v9-c/checkpoints/yolov9_traced.pt"
    torch.jit.save(traced_model, traced_model_path)
    print(f"\n💾 Traced model saved to: {traced_model_path}")
        
except Exception as e:
    print(f"❌ Tracing failed: {e}")

