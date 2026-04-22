import sys
sys.path.append('/mnt/c/Users/aroy/Projects/YOLO')

from pathlib import Path
from omegaconf import OmegaConf
from yolo.tools.data_loader import YoloDataset

# Hard-coded paths
config_path = '/mnt/c/Users/aroy/Projects/YOLO/yolo/config/dataset/doclaynet.yaml'
data_base_path = '/mnt/c/Users/aroy/Projects/DocLayout-YOLO/doclaynet_createml'

# Load dataset config
dataset_cfg = OmegaConf.load(config_path)
dataset_cfg.path = data_base_path

# Create complete data config
data_cfg = OmegaConf.create({
    'image_size': [640, 640],
    'batch_size': 1,
    'shuffle': True,
    'pin_memory': False,
    'cpu_num': 0,
    'source': '',
    'dynamic_shape': False,
    'data_augment': {}
})

print(f"Dataset config: {dataset_cfg}")

# Try loading validation
try:
    # Don't use DataConfig class, just pass OmegaConf directly
    val_dataset = YoloDataset(
        data_cfg=data_cfg,
        dataset_cfg=dataset_cfg,
        phase='validation'
    )
    
    print(f"\nValidation dataset loaded: {len(val_dataset)} samples")
    
    # Check if data was loaded
    print(f"Image paths: {len(val_dataset.img_paths)}")
    print(f"Bboxes: {len(val_dataset.bboxes)}")
    
    # Check for valid bboxes
    valid_count = 0
    empty_count = 0
    
    for i in range(len(val_dataset)):
        bbox = val_dataset.bboxes[i]
        if len(bbox) > 0 and bbox[0, 0] != -1:
            valid_count += 1
        else:
            empty_count += 1
            if empty_count <= 5:  # Show first 5 empty
                print(f"Empty bbox at index {i}: {val_dataset.img_paths[i]}")
    
    print(f"\nValid samples: {valid_count}")
    print(f"Empty samples: {empty_count}")
    
    # Check a valid sample
    for i in range(min(10, len(val_dataset))):
        bbox = val_dataset.bboxes[i]
        if len(bbox) > 0 and bbox[0, 0] != -1:
            print(f"\nFirst valid sample {i}:")
            print(f"  Image: {val_dataset.img_paths[i]}")
            print(f"  BBox: {bbox[0]}")
            break
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

