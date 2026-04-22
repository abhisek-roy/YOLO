import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_yolo_to_coco(images_dir, labels_dir, output_json):
    """Convert YOLO txt labels to COCO format."""
    
    categories = [
        {"id": 1, "name": "Caption"},
        {"id": 2, "name": "Footnote"},
        {"id": 3, "name": "Formula"},
        {"id": 4, "name": "List-item"},
        {"id": 5, "name": "Page-footer"},
        {"id": 6, "name": "Page-header"},
        {"id": 7, "name": "Picture"},
        {"id": 8, "name": "Section-header"},
        {"id": 9, "name": "Table"},
        {"id": 10, "name": "Text"},
        {"id": 11, "name": "Title"}
    ]
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 1
    image_id = 1
    
    # Get all image files
    image_files = list(Path(images_dir).glob("*.png"))
    
    for img_path in tqdm(image_files, desc="Converting"):
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Add image entry
        coco_format["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })
        
        # Read corresponding label file
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, w_norm, h_norm = map(float, parts)
                        
                        # Convert YOLO normalized to COCO absolute
                        x = (x_center - w_norm/2) * width
                        y = (y_center - h_norm/2) * height
                        w = w_norm * width
                        h = h_norm * height
                        
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id) + 1,  # COCO uses 1-indexed
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1
        
        image_id += 1
    
    # Save
    output_path = Path(output_json)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_format, f)
    
    print(f"✅ Converted {len(image_files)} images to {output_json}")

# Convert using existing YOLO labels
convert_yolo_to_coco(
    "../images/train",
    "../labels/train",
    "../annotations/instances_train.json"
)

convert_yolo_to_coco(
    "../images/validation",
    "../labels/validation",
    "../annotations/instances_val.json"
)

