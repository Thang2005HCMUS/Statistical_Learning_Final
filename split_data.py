import os
import shutil
import random

def split_yolo_dataset(input_dir, output_dir, split_ratio=0.8, seed=42):
    # Set the random seed for reproducibility
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Define input paths
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')

    # Allowed image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    # Get a list of all images
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
    
    # Shuffle the dataset
    random.shuffle(images)

    # Calculate the split index
    split_index = int(len(images) * split_ratio)

    # Split the lists
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Helper function to copy files
    def copy_files(image_list, subset):
        # Define output paths for this subset (train or val)
        out_images_dir = os.path.join(output_dir, subset, 'images')
        out_labels_dir = os.path.join(output_dir, subset, 'labels')

        # Create output directories if they don't exist
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_labels_dir, exist_ok=True)

        for img_name in image_list:
            # 1. Copy Image
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(out_images_dir, img_name)
            shutil.copy(src_img, dst_img)

            # 2. Copy Label
            # Get the filename without the extension and add .txt
            base_name = os.path.splitext(img_name)[0]
            label_name = base_name + '.txt'
            
            src_label = os.path.join(labels_dir, label_name)
            dst_label = os.path.join(out_labels_dir, label_name)

            # Check if the label exists before copying (handles images with no labels)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                print(f"Warning: No label found for {img_name}")

    print(f"Total images found: {len(images)}")
    print("Copying Training files...")
    copy_files(train_images, 'train')
    
    print("Copying Validation files...")
    copy_files(val_images, 'val')
    
    print(f"\nDone! Split Output -> Train: {len(train_images)} | Val: {len(val_images)}")

# --- HOW TO USE ---
# Change these paths to match your actual folder locations
INPUT_FOLDER = './data/Fisheye8K/train'   # Folder containing 'images' and 'labels'
OUTPUT_FOLDER = './data/Fisheye8K_train_split' # Where you want the split dataset to be saved

split_yolo_dataset(INPUT_FOLDER, OUTPUT_FOLDER, split_ratio=0.8, seed=42)