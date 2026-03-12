import sys
import random
import numpy as np
import json
import torch
from PIL import Image, ImageDraw

sys.path.append('/usr/local/aroy/Projects/YOLO')

from yolo import create_converter, PostProcess, NMSConfig, draw_bboxes, AugmentationComposer
from omegaconf import OmegaConf

# ========== CONFIGURATION ==========

class_list = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 
              'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']

IMAGE_SIZE = (1024, 1024)
device = torch.device("cpu")

nms_config = NMSConfig(
    min_confidence=0.25,
    min_iou=0.45,
    max_bbox=100
)

# ========== LOAD ANNOTATIONS ==========

print("📋 Loading ground truth annotations...")
with open('/usr/local/aroy/Projects/YOLO/doclaynet/images/annotations/instances_val.json', 'r') as f:
    coco_data = json.load(f)

# Create lookup dictionaries
image_id_to_info = {img['id']: img for img in coco_data['images']}
image_filename_to_id = {img['file_name']: img['id'] for img in coco_data['images']}

print(f"  Loaded {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

# ========== LOAD MODEL ==========

print("\n📥 Loading model...")
torch_model = torch.jit.load("/usr/local/aroy/Projects/YOLO/runs/train/v9-c/checkpoints/yolov9_traced.pt")
torch_model.eval()

model_cfg = OmegaConf.load('/usr/local/aroy/Projects/YOLO/yolo/config/model/v9-c.yaml')
converter = create_converter(model_cfg.name, torch_model, model_cfg.anchor, IMAGE_SIZE, device)
post_process = PostProcess(converter, nms_config)
transform = AugmentationComposer([])

print("✅ Model loaded!")

# ========== SELECT TEST IMAGE ==========

# Pick first image from validation set
test_filename = random.choice(coco_data['images'])['file_name']
print(f"\n📸 Test image: {test_filename}")

# Or specify a particular image:
# test_filename = "YOUR_IMAGE_NAME.png"

image_path = f"/usr/local/aroy/Projects/YOLO/doclaynet/images/images/val/{test_filename}"

# ========== LOAD GROUND TRUTH FOR THIS IMAGE ==========

image_id = image_filename_to_id[test_filename]
image_info = image_id_to_info[image_id]

# Get all annotations for this image
gt_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

print(f"  Ground truth boxes: {len(gt_annotations)}")

# Convert COCO annotations to draw_bboxes format
# COCO: [x, y, width, height] → [class_id, x_min, y_min, x_max, y_max]
gt_bboxes = []
for ann in gt_annotations:
    x, y, w, h = ann['bbox']
    class_id = ann['category_id'] - 1  # COCO categories are 1-indexed
    gt_bboxes.append([class_id, x, y, x + w, y + h])

# ========== RUN INFERENCE ==========

print("\n🔮 Running inference...")
original_image = Image.open(image_path).convert("RGB")
print(f"  Image size: {original_image.size}")

# Transform for model (will resize 1025→1024)
resized_image = original_image.resize((1024, 1024), Image.LANCZOS)
img_array = np.array(resized_image).astype(np.float32) / 255.0
image_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
rev_tensor = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]).to(device)

with torch.no_grad():
    predictions = torch_model(image_tensor)
    # Restructure: 9 flat tensors → 3 scales of (cls, box_dist, box_coord)
    predictions_restructured = [
        (predictions[0], predictions[1], predictions[2]),  # Scale 0
        (predictions[3], predictions[4], predictions[5]),  # Scale 1
        (predictions[6], predictions[7], predictions[8])   # Scale 2
    ]
    predictions_dict = {"Main": predictions_restructured}
    pred_bboxes = post_process(predictions_dict, rev_tensor)

print(f"  Predictions: {len(pred_bboxes[0])}")

# ========== VISUALIZE ==========

print("\n🎨 Creating visualizations...")

# 1. Ground truth only
gt_image = draw_bboxes(original_image.copy(), [gt_bboxes], idx2label=class_list)
# gt_image.save("ground_truth.png")

# 2. Predictions only
pred_image = draw_bboxes(original_image.copy(), pred_bboxes, idx2label=class_list)
# pred_image.save("predictions.png")

# 3. Side-by-side comparison
comparison = Image.new('RGB', (original_image.width * 2, original_image.height))
comparison.paste(gt_image, (0, 0))
comparison.paste(pred_image, (original_image.width, 0))

# Add labels
draw = ImageDraw.Draw(comparison)
draw.text((10, 10), "GROUND TRUTH", fill='white')
draw.text((original_image.width + 10, 10), "PREDICTIONS", fill='white')

comparison.save("comparison.png")

print("\n💾 Saved:")
print("  - ground_truth.png")
print("  - predictions.png")
print("  - comparison.png")

print("\n📊 Summary:")
print(f"  Ground truth boxes: {len(gt_bboxes)}")
print(f"  Predicted boxes: {len(pred_bboxes[0])}")

