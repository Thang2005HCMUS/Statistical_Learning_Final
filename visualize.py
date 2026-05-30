import cv2
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# Define colors and class names
class_colors = {
    0: (255, 0, 0),    # Red for bus
    1: (0, 255, 0),    # Green for bike
    2: (0, 0, 255),    # Blue for car
    3: (255, 255, 0),  # Yellow for pedestrian
    4: (255, 0, 255),  # Purple for truck
}

class_names = {
    0: "bus",
    1: "bike",
    2: "bike",
    3: "pedes",
    4: "truck",
}

threshold = [0.6, 0.6, 0.37, 0.6, 0.6]

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO bounding boxes on images")
    parser.add_argument("--images", type=Path, required=True, help="Folder containing input images (png/jpg/jpeg)")
    parser.add_argument("--labels", type=Path, required=True, help="Folder containing YOLO label files (.txt)")
    parser.add_argument("--output", type=Path, required=True, help="Folder to save output images with boxes")
    return parser.parse_args()

def main():
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image_exts = [".png", ".jpg", ".jpeg"]

    image_files = [f for f in args.images.iterdir() if f.suffix.lower() in image_exts]

    for image_path in tqdm(image_files, desc="Processing images"):
        label_path = args.labels / (image_path.stem + ".txt")
        if not label_path.exists():
            continue  # Skip if no label

        image = cv2.imread(str(image_path))
        if image is None:
            continue  # Skip unreadable images
        
        h, w, _ = image.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()
        flag = 1
        for line in lines:
            data = line.strip().split()
            
            class_id = int(data[0])
            # if (len(data) <= 5): 
            #     flag = 0
            #     break
            x_center, y_center, width, height = map(float, data[1:])

            # if score < threshold[class_id]: continue

            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            color = class_colors.get(class_id, (255, 255, 255))
            label = class_names.get(class_id, str(class_id))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(image, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        output_path = args.output / image_path.name
        if flag:
            cv2.imwrite(str(output_path), image)

if __name__ == "__main__":
    main()
