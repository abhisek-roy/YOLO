import sys
sys.path.append('/mnt/c/Users/aroy/Projects/YOLO')

from pathlib import Path
from yolo.utils.dataset_utils import locate_label_paths

data_base_path = Path('/mnt/c/Users/aroy/Projects/DocLayout-YOLO/doclaynet_createml')

# Check what data type is detected for each phase
for phase in ['train', 'validation']:
    print(f"\n{phase}:")
    labels_path, data_type = locate_label_paths(data_base_path, phase)
    print(f"  Labels path: {labels_path}")
    print(f"  Data type: {data_type}")
    
    # Check if txt file list exists
    file_list = data_base_path / f"{phase}.txt"
    print(f"  {phase}.txt exists: {file_list.exists()}")
    
    # Check actual label files
    if data_type == "txt":
        label_dir = data_base_path / "labels" / phase
        if label_dir.exists():
            txt_files = list(label_dir.glob("*.txt"))
            print(f"  Found {len(txt_files)} txt files")
            if txt_files:
                # Check content of first file
                with open(txt_files[0]) as f:
                    first_line = f.readline().strip()
                    print(f"  First label line: {first_line}")
                    values = first_line.split()
                    print(f"  Number of values: {len(values)}")

