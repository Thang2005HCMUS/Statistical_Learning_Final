import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Import module model của bạn
from models import fasterrcnn_resnet50_fpn

# Danh sách class (Index 0 là background)
CLASSES = ['background', 'bus', 'bike', 'car', 'pedestrian', 'truck']
IMGSZ = 640

class VideoInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Faster R-CNN Video Inference")
        self.root.geometry("1100x750")
        
        # Biến hệ thống
        self.model = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Biến xử lý Video
        self.cap = None
        self.is_playing = False
        
        self.setup_ui()

    def setup_ui(self):
        # ================= FRAME TRÁI (Bảng điều khiển) =================
        left_frame = tk.Frame(self.root, width=300, bg="#f4f4f4")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 1. Load Trọng số (Weights)
        tk.Label(left_frame, text="1. Tải Mô hình", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        tk.Button(left_frame, text="Chọn file .pth (Weights)", command=self.load_model, width=30).pack()
        self.lbl_weights = tk.Label(left_frame, text="Chưa tải mô hình...", fg="red", bg="#f4f4f4", wraplength=250)
        self.lbl_weights.pack(pady=5)

        # 2. Chọn Video
        tk.Label(left_frame, text="2. Nguồn Video", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(20, 2))
        tk.Button(left_frame, text="Chọn file Video (.mp4, .avi...)", command=self.load_video, width=30).pack()
        self.lbl_video = tk.Label(left_frame, text="Chưa chọn video...", fg="blue", bg="#f4f4f4", wraplength=250)
        self.lbl_video.pack(pady=5)

        # 3. Điều khiển Video
        tk.Label(left_frame, text="3. Điều khiển", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(20, 2))
        control_frame = tk.Frame(left_frame, bg="#f4f4f4")
        control_frame.pack(pady=5)
        
        self.btn_play = tk.Button(control_frame, text="▶ Phát / Tạm dừng", command=self.toggle_play, width=15, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(control_frame, text="⏹ Dừng hẳn", command=self.stop_video, width=10, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # 4. Thanh trượt Threshold
        tk.Label(left_frame, text="Ngưỡng tin cậy (Confidence)", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(20, 2))
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.slider = tk.Scale(left_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, variable=self.conf_threshold, bg="#f4f4f4")
        self.slider.pack(fill=tk.X, padx=20)

        # ================= FRAME PHẢI (Hiển thị Video) =================
        self.right_frame = tk.Frame(self.root, bg="black")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = tk.Label(self.right_frame, text="Màn hình hiển thị Video\n(Resize về 640x640)", bg="black", fg="white", font=("Arial", 14))
        self.video_label.pack(expand=True)

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Chọn trọng số mô hình", filetypes=[("PyTorch Weights", "*.pth *.pt")])
        if filepath:
            self.lbl_weights.config(text="Đang nạp mô hình, chờ xíu nhé...", fg="orange")
            self.root.update()
            
            try:
                self.model = fasterrcnn_resnet50_fpn.create_model(num_classes=len(CLASSES), coco_model=False)
                
                checkpoint = torch.load(filepath, map_location=self.device)
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                    
                self.model.load_state_dict(new_state_dict)
                self.model.to(self.device)
                self.model.eval()
                
                self.lbl_weights.config(text=f"Đã nạp model trên: {self.device}", fg="green")
                messagebox.showinfo("Thành công", "Đã nạp model! Giờ hãy chọn Video.")
            except Exception as e:
                self.lbl_weights.config(text="Lỗi nạp mô hình!", fg="red")
                messagebox.showerror("Lỗi", f"Không thể load model:\n{str(e)}")

    def load_video(self):
        if self.model is None:
            messagebox.showwarning("Nhắc nhở", "Phải nạp Model trước khi mở Video bạn nhé!")
            return
            
        filepath = filedialog.askopenfilename(title="Chọn Video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if filepath:
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(filepath)
            self.lbl_video.config(text=os.path.basename(filepath), fg="green")
            
            # Kích hoạt các nút bấm
            self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.NORMAL)
            
            # Reset trạng thái
            self.is_playing = False
            
            # Đọc thử 1 frame đầu tiên để hiển thị lên màn hình
            ret, frame = self.cap.read()
            if ret:
                self.process_single_frame(frame)

    def toggle_play(self):
        if self.cap is None or not self.cap.isOpened():
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ Tạm dừng")
            self.play_video_loop() # Kích hoạt vòng lặp chạy video
        else:
            self.btn_play.config(text="▶ Tiếp tục")

    def stop_video(self):
        self.is_playing = False
        self.btn_play.config(text="▶ Phát / Tạm dừng", state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        self.lbl_video.config(text="Đã dừng Video", fg="blue")
        self.video_label.config(image='', text="Màn hình hiển thị Video\n(Resize về 640x640)")

    def play_video_loop(self):
        if not self.is_playing or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.process_single_frame(frame)
            # Hàm root.after sẽ gọi lại chính hàm này sau 1ms, tạo thành vòng lặp 
            # (Tốc độ thực tế phụ thuộc vào việc process_single_frame tốn bao nhiêu thời gian)
            self.root.after(1, self.play_video_loop)
        else:
            # Hết video
            self.stop_video()
            messagebox.showinfo("Hoàn thành", "Đã phát hết Video!")

    def process_single_frame(self, frame_bgr):
        """Hàm bóc tách lõi: Xử lý 1 frame duy nhất qua model"""
        # Resize về 640x640 để khớp hoàn toàn với những gì model được học
        frame_resized = cv2.resize(frame_bgr, (IMGSZ, IMGSZ))
        draw_img = frame_resized.copy()

        # Tiền xử lý cho Tensor
        img_rgb_for_model = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb_for_model.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Chạy inference
        with torch.inference_mode():
            outputs = self.model(img_tensor)
        
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        
        threshold = self.conf_threshold.get()

        # Vẽ Bounding Box
        for box, score, label in zip(boxes, scores, labels):
            if score >= threshold:
                xmin, ymin, xmax, ymax = map(int, box)
                cls_name = CLASSES[label] if label < len(CLASSES) else f"CLS_{label}"
                
                # Khung màu Vàng cam cho nổi trên video
                cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), (0, 165, 255), 2)
                # Đổ bóng nền text cho dễ đọc
                cv2.rectangle(draw_img, (xmin, ymin - 20), (xmin + 100, ymin), (0, 165, 255), -1)
                cv2.putText(draw_img, f"{cls_name} {score:.2f}", (xmin + 3, ymin - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Chuyển lên Tkinter
        draw_img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(draw_img_rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.video_label.config(image=self.tk_image, text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoInferenceApp(root)
    root.mainloop()