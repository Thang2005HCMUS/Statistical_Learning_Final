
import os
import yaml
from ultralytics import YOLO, RTDETR
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model_path = "rtdetr-l.pt"
model = RTDETR(model_path)

# Load YAML argumentsss
model.train(data='data.yaml', epochs=80, batch=8, imgsz=640, name='rtdetr-l', save_period=5, exist_ok=True, workers=32)
