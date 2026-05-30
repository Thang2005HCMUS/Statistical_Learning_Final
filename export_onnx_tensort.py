from ultralytics import YOLO
import os

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

# Load trained model
model_path = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/Track4_AIC_FishEyecamera/tvh/yolo/weight10s_group_indraeye/train9/weights/best.pt'
model = YOLO(model_path)

# Export to TensorRT with FP16
export_result = model.export(format="engine", half=True)

# Confirm saved path
print(f"FP16 TensorRT engine saved at: {export_result}")
