import os
from pathlib import Path
from collections import defaultdict

# Update these paths to match your dataset structure
BASE_PATH = "./dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
VALID_PATH = os.path.join(BASE_PATH, "valid")
TEST_PATH = os.path.join(BASE_PATH, "test")

def get_classes_from_split(split_path):
    """Extract all unique class folders/labels from a dataset split."""
    classes = set()
    
    # Check if using folder structure (images organized by class)
    if os.path.exists(split_path):
        items = os.listdir(split_path)
        # Look for subdirectories (class folders)
        for item in items:
            item_path = os.path.join(split_path, item)
            if os.path.isdir(item_path):
                classes.add(item)
    
    return classes

def get_classes_from_labels(split_path):
    """Extract classes from YOLO label files."""
    classes = set()
    labels_path = os.path.join(split_path, "labels")
    
    if os.path.exists(labels_path):
        for label_file in Path(labels_path).glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        classes.add(class_id)
    
    return classes

# Get classes from each split
print("Analyzing dataset splits...\n")

train_classes = get_classes_from_split(TRAIN_PATH)
valid_classes = get_classes_from_split(VALID_PATH)
test_classes = get_classes_from_split(TEST_PATH)

# If folder structure didn't work, try label files
if not train_classes:
    print("Trying label file analysis...\n")
    train_classes = get_classes_from_labels(BASE_PATH.replace("/train", ""))
    valid_classes = get_classes_from_labels(BASE_PATH.replace("/valid", ""))
    test_classes = get_classes_from_labels(BASE_PATH.replace("/test", ""))

print(f"Train set: {len(train_classes)} classes")
print(f"Valid set: {len(valid_classes)} classes")
print(f"Test set: {len(test_classes)} classes")
print()

# Find missing classes
missing_in_valid = train_classes - valid_classes
missing_in_test = train_classes - test_classes

print("="*60)
print("MISSING CLASSES ANALYSIS")
print("="*60)

if missing_in_valid:
    print(f"\n❌ Classes in TRAIN but missing in VALID ({len(missing_in_valid)}):")
    print(f"   {sorted(missing_in_valid)}")
else:
    print("\n✅ All training classes present in validation set")

if missing_in_test:
    print(f"\n❌ Classes in TRAIN but missing in TEST ({len(missing_in_test)}):")
    print(f"   {sorted(missing_in_test)}")
else:
    print("\n✅ All training classes present in test set")

# Show all classes for reference
print(f"\n{'='*60}")
print("ALL CLASSES")
print("="*60)
print(f"\nTrain classes: {sorted(train_classes)}")
print(f"\nValid classes: {sorted(valid_classes)}")
print(f"\nTest classes: {sorted(test_classes)}")

# Count images per class if using folder structure
if os.path.isdir(TRAIN_PATH):
    print(f"\n{'='*60}")
    print("IMAGE COUNT PER CLASS (Train set)")
    print("="*60)
    for cls in sorted(train_classes):
        cls_path = os.path.join(TRAIN_PATH, str(cls))
        if os.path.exists(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            marker = " ⚠️ MISSING IN VALID" if cls in missing_in_valid else ""
            marker += " ⚠️ MISSING IN TEST" if cls in missing_in_test else ""
            print(f"  Class {cls}: {count} images{marker}")