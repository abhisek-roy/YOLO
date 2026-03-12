import torch
import coremltools as ct

print("📥 Loading traced model...")
traced_model = torch.jit.load("/usr/local/aroy/Projects/YOLO/runs/train/v9-c/checkpoints/yolov9_traced.pt")
traced_model.eval()
print("✅ Traced model loaded!")

# ========== CONVERT TO COREML ==========

print("\n🔄 Converting to CoreML...")

# Define input shape (1024x1024 as per your training)
input_shape = (1, 3, 1024, 1024)

# Define image input with preprocessing
# Adjust scale and bias based on your training normalization
scale = 1.0 / 255.0  # Simple [0,1] normalization
bias = [0, 0, 0]     # No bias

image_input = ct.ImageType(
    name="image",
    shape=input_shape,
    scale=scale,
    bias=bias,
    color_layout=ct.colorlayout.RGB
)

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[image_input],
    compute_units=ct.ComputeUnit.ALL,  # Use CPU, GPU, and Neural Engine
    minimum_deployment_target=ct.target.iOS15,
)

print("✅ Conversion successful!")

# Save the model
output_path = "/usr/local/aroy/Projects/YOLO/runs/train/v9-c/checkpoints/YOLOv9_DocLayNet.mlpackage"
mlmodel.save(output_path)
print(f"\n💾 Model saved to: {output_path}")

print(f"\n📊 Model info:")
print(f"  Input: {mlmodel.get_spec().description.input[0].name}")
print(f"  Number of outputs: {len(mlmodel.get_spec().description.output)}")

