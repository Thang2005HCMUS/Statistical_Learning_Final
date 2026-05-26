import os
import gc
import cv2
import torch
import base64
import numpy as np
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO, RTDETR
import torchvision
from typing import List
import uuid

app = FastAPI(title="Object Detection API")

# Cấu hình CORS để Frontend (React) có thể gọi được API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def cleanup_temp_file(path: str):
    """Hàm chạy ngầm để xóa file tạm"""
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Đã dọn dẹp file tạm: {path}")
    except Exception as e:
        print(f"[ERROR] Không thể xóa file {path}: {e}")
# ==========================================
# 1. QUẢN LÝ MODEL (SINGLETON PATTERN)
# ==========================================
class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.current_model_key = None
            cls._instance.model = None
            cls._instance.model_type = None # 'ultralytics' hoặc 'faster-rcnn'
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._instance

    def load_model(self, model_name: str, weight_type: str):
        model_key = f"{model_name}_{weight_type}"
        
        # Nếu model đã được load rồi thì dùng luôn (Singleton)
        if self.current_model_key == model_key:
            print(f"[INFO] Reusing already loaded model: {model_key}")
            return

        # Nếu đang có một model khác, tiến hành giải phóng bộ nhớ
        if self.model is not None:
            print(f"[INFO] Unloading model: {self.current_model_key}")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        print(f"[INFO] Loading new model: {model_key} ...")
        
        # Xác định đuôi file
        ext = "pth" if model_name == "faster-rcnn" else "pt"
        weight_path = os.path.join("../models", model_name, f"{weight_type}.{ext}")

        if not os.path.exists(weight_path):
            raise HTTPException(status_code=404, detail=f"Model weight not found at {weight_path}")

        # Nạp model tùy theo loại
        if "yolo" in model_name:
            self.model = YOLO(weight_path)
            self.model_type = 'ultralytics'
        elif "rtdetr" in model_name:
            self.model = RTDETR(weight_path)
            self.model_type = 'ultralytics'
        elif "faster-rcnn" in model_name:
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from collections import OrderedDict
            
            # --- LƯU Ý QUAN TRỌNG: CẬP NHẬT SỐ CLASS ---
            # Sửa NUM_CLASSES đúng với lúc bạn train (bao gồm cả class 0 là background)
            # Ví dụ: Data có 3 classes -> NUM_CLASSES = 4
            NUM_CLASSES = 6

            # 1. Khởi tạo model và thay thế Head giống hệt lúc eval
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

            # 2. Tải file weights/checkpoint
            checkpoint = torch.load(weight_path, map_location=self.device)
            
            # Lấy dict chứa trọng số
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            # 3. Loại bỏ chữ 'module.' khỏi các key (xử lý DataParallel)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v

            # 4. Load weights và nạp vào thiết bị (CPU/GPU)
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.model_type = 'faster-rcnn'
        else:
            raise HTTPException(status_code=400, detail="Unknown model name")

        self.current_model_key = model_key
        print(f"[INFO] Successfully loaded: {model_key}")

    def predict(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Hàm dự đoán chung, nhận vào list ảnh (numpy) và trả về list ảnh đã vẽ box"""
        results_images = []
        
        if self.model_type == 'ultralytics':
            # Ultralytics tự động hỗ trợ batch inference
            results = self.model(images, verbose=False)
            for r in results:
                results_images.append(r.plot()) # plot() trả về numpy array đã vẽ box
                
        elif self.model_type == 'faster-rcnn':
            # Custom inference cho Faster R-CNN
            transform = torchvision.transforms.ToTensor()
            tensors = [transform(img).to(self.device) for img in images]
            with torch.no_grad():
                predictions = self.model(tensors)
            
            for img, pred in zip(images, predictions):
                img_copy = img.copy()
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                for box, score in zip(boxes, scores):
                    if score > 0.5: # Ngưỡng confidence
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_copy, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                results_images.append(img_copy)
                
        return results_images

# Khởi tạo Singleton
model_manager = ModelManager()

# ==========================================
# CÁC HÀM TIỆN ÍCH (HELPER FUNCTIONS)
# ==========================================
def read_image_file(file: bytes) -> np.ndarray:
    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def image_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

# ==========================================
# 2. CÁC API ENDPOINTS
# ==========================================

@app.post("/api/detect/single-image")
async def detect_single_image(
    model_name: str = Form(...),
    weight_type: str = Form(...),
    file: UploadFile = File(...)
):
    """API 1: Xử lý 1 ảnh đơn"""
    # Nạp/Kiểm tra model
    model_manager.load_model(model_name, weight_type)
    
    # Đọc ảnh
    contents = await file.read()
    img = read_image_file(contents)
    
    # Dự đoán
    res_img = model_manager.predict([img])[0]
    
    # Trả về dạng file ảnh trực tiếp
    _, encoded_img = cv2.imencode('.jpg', res_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@app.post("/api/detect/batch-images")
async def detect_batch_images(
    model_name: str = Form(...),
    weight_type: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """API 2: Xử lý nhiều ảnh (Batch size = 4)"""
    model_manager.load_model(model_name, weight_type)
    
    # Đọc tất cả ảnh
    images = []
    for f in files:
        contents = await f.read()
        images.append(read_image_file(contents))
        
    # Xử lý theo batch
    batch_size = 4
    results_base64 = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        batch_results = model_manager.predict(batch)
        # Convert sang Base64 để trả về JSON (Frontend dễ render bằng mảng src)
        results_base64.extend([image_to_base64(res) for res in batch_results])
        
    return JSONResponse(content={"results": results_base64})


@app.post("/api/detect/video")
async def detect_video(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    weight_type: str = Form(...),
    file: UploadFile = File(...)
):
    """API 3: Xử lý video"""
    model_manager.load_model(model_name, weight_type)
    
    # Tạo file tạm
    session_id = str(uuid.uuid4())
    input_vid_path = f"temp_in_{session_id}.mp4"
    output_vid_path = f"temp_out_{session_id}.webm" # Dùng webm (VP8) để trình duyệt web hiển thị trực tiếp dễ dàng
    
    # Lưu video upload xuống đĩa
    with open(input_vid_path, "wb") as f:
        f.write(await file.read())
        
    # Đọc video bằng OpenCV
    cap = cv2.VideoCapture(input_vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Cấu hình writer (vp8 cho webm hoặc avc1 cho mp4)
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_vid_path, fourcc, fps, (width, height))
    
    frames_batch = []
    batch_size = 4
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_batch.append(frame)
        
        # Nếu đủ batch size thì predict
        if len(frames_batch) == batch_size:
            res_frames = model_manager.predict(frames_batch)
            for rf in res_frames:
                out.write(rf)
            frames_batch = []
            
    # Xử lý nốt những frame còn sót lại
    if len(frames_batch) > 0:
        res_frames = model_manager.predict(frames_batch)
        for rf in res_frames:
            out.write(rf)
            
    cap.release()
    out.release()
    
    # Xóa file input tạm
    if os.path.exists(input_vid_path):
        os.remove(input_vid_path)
    background_tasks.add_task(cleanup_temp_file, output_vid_path)
    # Trả về file video output (sử dụng background task để xóa file output sau khi gửi nếu cần, ở đây trả về trực tiếp)
    return FileResponse(output_vid_path, media_type="video/webm", filename="result.webm")

# Lệnh khởi chạy server (dùng trong terminal):
# uvicorn main:app --reload --port 8000