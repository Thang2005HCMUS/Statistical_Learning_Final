from ultralytics import RTDETR, YOLO, YOLOv10
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# CONFIG---------------------
model_path = './runs/detect/rtdetr-l/weights/best.pt'
model = RTDETR(model_path)
name = 'rtdetr-l'

# model_path = './runs/detect/yolov10s/weights/best.pt'
# model = YOLO(model_path)
# name = 'yolov10s'

# --------- Tracking with FPS ---------
folder = Path("./data/Fisheye8K/test/images")

start = time.time()

results = model.predict(
    folder,
    task='detect',
    save=True,
    conf=0.5,
    augment=False,
    agnostic_nms=False,
    imgsz=640,
    show_labels=False,
    classes= [0,1,2,3,4],
    project='test',
    name=name,
    batch=8,
    exist_ok=True
)

end = time.time()
total_time = end - start
image_count = len(list(folder.glob("*.png")))
fps = image_count / total_time if total_time > 0 else 0

print(f"\nProcessed {image_count} images in {total_time:.3f} seconds. FPS: {fps:.3f}")
