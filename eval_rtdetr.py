from ultralytics import YOLO, RTDETR

# Load the model
model_path = './runs/detect/rtdetr-l/weights/best.pt'
model = RTDETR(model_path)

# Customize validation settings
results = model.val(data="data_val.yaml", imgsz=640, project='eval_rtdetr-l', exist_ok=True)

print("mAP (50-95):", results.box.map)  # mAP50-95
print("mAP50:", results.box.map50)  # mAP50
print("mAP75:", results.box.map75)  # mAP75
print("mAP Scores (per IoU):", results.box.maps)
mean_f1_score = sum(results.box.f1) / len(results.box.f1)
print(f"Mean F1 Score: {mean_f1_score:.4f}")

print("Class indices with average precision:", results.ap_class_index)
print("Average precision:", results.box.ap)
print("Average precision at IoU=0.50:", results.box.ap50)
print("F1 score:", results.box.f1)

print("Mean average precision:", results.box.map)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)
print("Mean average precision for different IoU thresholds:", results.box.maps)
print("Mean precision:", results.box.mp)
print("Mean recall:", results.box.mr)
print("Precision:", results.box.p)
print("Recall:", results.box.r)
