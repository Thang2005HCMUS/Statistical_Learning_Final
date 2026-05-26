import os
import glob
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Import module model của bạn
from models import fasterrcnn_resnet50_fpn

# Danh sách class (Index 0 là background)
CLASSES = ['background', 'bus', 'bike', 'car', 'pedestrian', 'truck']
IMGSZ = 640

class CompareViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Ground Truth vs Faster R-CNN Prediction (Side-by-Side)")
        # Tăng kích thước cửa sổ để chứa 2 ảnh 640x640 cạnh nhau
        self.root.geometry("1650x800") 
        
        self.img_dir = ""
        self.label_dir = ""
        self.weights_path = ""
        self.image_paths = []
        
        # Biến Singleton chứa Model
        self.model = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.setup_ui()

    def setup_ui(self):
        # ================= FRAME TRÁI (Bảng điều khiển) =================
        left_frame = tk.Frame(self.root, width=320, bg="#f4f4f4")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 1. Load Trọng số (Weights)
        tk.Label(left_frame, text="1. Tải Mô hình (Bắt buộc)", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        tk.Button(left_frame, text="Chọn file .pth (Weights)", command=self.load_model, width=30).pack()
        self.lbl_weights = tk.Label(left_frame, text="Chưa tải mô hình...", fg="red", bg="#f4f4f4", wraplength=300)
        self.lbl_weights.pack(pady=5)

        # 2. Chọn Thư mục
        tk.Label(left_frame, text="2. Dữ liệu", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        tk.Button(left_frame, text="Chọn thư mục Ảnh", command=self.browse_img_dir, width=30).pack(pady=2)
        tk.Button(left_frame, text="Chọn thư mục Nhãn (.txt)", command=self.browse_label_dir, width=30).pack(pady=2)
        self.lbl_data_status = tk.Label(left_frame, text="Chưa chọn đủ dữ liệu", fg="blue", bg="#f4f4f4")
        self.lbl_data_status.pack(pady=5)

        # 3. Thanh trượt Threshold
        tk.Label(left_frame, text="3. Ngưỡng tin cậy (Confidence)", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.slider = tk.Scale(left_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, variable=self.conf_threshold, command=self.on_slider_change, bg="#f4f4f4")
        self.slider.pack(fill=tk.X, padx=20)

        # 4. Danh sách ảnh
        tk.Label(left_frame, text="Danh sách ảnh:", bg="#f4f4f4").pack(anchor=tk.W, pady=(15, 0))
        list_frame = tk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.scrollbar = tk.Scrollbar(list_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(list_frame, yscrollcommand=self.scrollbar.set, width=35)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.listbox.yview)
        
        self.listbox.bind('<<ListboxSelect>>', self.on_select_image)

        # ================= FRAME PHẢI (Hiển thị 2 Ảnh) =================
        self.right_frame = tk.Frame(self.root, bg="#2c3e50")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame chứa tiêu đề
        title_frame = tk.Frame(self.right_frame, bg="#2c3e50")
        title_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        lbl_gt_title = tk.Label(title_frame, text="🟩 Nhãn gốc (Ground Truth)", fg="lime", bg="#2c3e50", font=("Arial", 14, "bold"), width=45)
        lbl_gt_title.pack(side=tk.LEFT, expand=True)
        
        lbl_pred_title = tk.Label(title_frame, text="🟥 Mô hình Dự đoán (Prediction)", fg="#ff4757", bg="#2c3e50", font=("Arial", 14, "bold"), width=45)
        lbl_pred_title.pack(side=tk.RIGHT, expand=True)

        # Frame chứa 2 ảnh
        img_frame = tk.Frame(self.right_frame, bg="#2c3e50")
        img_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label_gt = tk.Label(img_frame, text="Ảnh Ground Truth sẽ hiển thị ở đây", bg="black", fg="white")
        self.image_label_gt.pack(side=tk.LEFT, padx=5, expand=True)

        self.image_label_pred = tk.Label(img_frame, text="Ảnh Prediction sẽ hiển thị ở đây", bg="black", fg="white")
        self.image_label_pred.pack(side=tk.RIGHT, padx=5, expand=True)
        
        # Biến lưu trữ đường dẫn ảnh đang chọn
        self.current_img_path = None
        self.current_label_path = None

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Chọn trọng số mô hình", filetypes=[("PyTorch Weights", "*.pth *.pt")])
        if filepath:
            self.lbl_weights.config(text="Đang nạp mô hình, vui lòng chờ...", fg="orange")
            self.root.update()
            
            try:
                # Khởi tạo model
                self.model = fasterrcnn_resnet50_fpn.create_model(num_classes=len(CLASSES), coco_model=False)
                
                # Sửa lỗi 'module.' trong state_dict
                checkpoint = torch.load(filepath, map_location=self.device)
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                    
                self.model.load_state_dict(new_state_dict)
                self.model.to(self.device)
                self.model.eval() # Chuyển sang chế độ evaluation
                
                self.lbl_weights.config(text=f"Đã nạp: {os.path.basename(filepath)}", fg="green")
                messagebox.showinfo("Thành công", "Load model thành công!")
            except Exception as e:
                self.lbl_weights.config(text="Lỗi nạp mô hình!", fg="red")
                messagebox.showerror("Lỗi", f"Không thể load model:\n{str(e)}")

    def browse_img_dir(self):
        folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        if folder:
            self.img_dir = folder
            self.load_image_list()
            
            # Tự đoán thư mục nhãn
            if not self.label_dir:
                guess_label_dir = folder.replace('images', 'labels')
                if os.path.exists(guess_label_dir) and guess_label_dir != folder:
                    self.label_dir = guess_label_dir
                else:
                    self.label_dir = folder
            self.update_data_status()

    def browse_label_dir(self):
        folder = filedialog.askdirectory(title="Chọn thư mục chứa nhãn (.txt)")
        if folder:
            self.label_dir = folder
            self.update_data_status()

    def update_data_status(self):
        if self.img_dir and self.label_dir:
            self.lbl_data_status.config(text="Đã sẵn sàng", fg="green")
        else:
            self.lbl_data_status.config(text="Chưa chọn đủ dữ liệu", fg="blue")

    def load_image_list(self):
        self.listbox.delete(0, tk.END)
        self.image_paths = glob.glob(os.path.join(self.img_dir, '*.png')) + \
                           glob.glob(os.path.join(self.img_dir, '*.jpg')) + \
                           glob.glob(os.path.join(self.img_dir, '*.jpeg'))
        for path in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(path))

    def on_select_image(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        index = selection[0]
        self.current_img_path = self.image_paths[index]
        base_name = os.path.basename(self.current_img_path).rsplit('.', 1)[0]
        self.current_label_path = os.path.join(self.label_dir, base_name + '.txt')
        self.process_and_display()

    def on_slider_change(self, event):
        if self.current_img_path: 
            self.process_and_display()

    def process_and_display(self):
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Bạn phải nạp Mô hình (Bước 1) trước!")
            return

        # 1. Đọc và Resize ảnh gốc
        image_bgr = cv2.imread(self.current_img_path)
        if image_bgr is None: return
        image_resized = cv2.resize(image_bgr, (IMGSZ, IMGSZ))
        
        # Tạo 2 bản sao độc lập cho GT và Prediction
        img_gt = image_resized.copy()
        img_pred = image_resized.copy()

        # 2. VẼ GROUND TRUTH (Màu Xanh Lá - Lên img_gt)
        if os.path.exists(self.current_label_path):
            with open(self.current_label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        xmin, ymin = int((cx - w / 2) * IMGSZ), int((cy - h / 2) * IMGSZ)
                        xmax, ymax = int((cx + w / 2) * IMGSZ), int((cy + h / 2) * IMGSZ)
                        
                        try: cls_name = CLASSES[int(class_id) + 1]
                        except: cls_name = f"CLS_{int(class_id)}"
                        
                        cv2.rectangle(img_gt, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(img_gt, f"GT: {cls_name}", (xmin, max(ymin - 8, 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. DỰ ĐOÁN VỚI MODEL (Màu Đỏ - Lên img_pred)
        img_rgb_for_model = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb_for_model.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(img_tensor)
        
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        
        threshold = self.conf_threshold.get()

        for box, score, label in zip(boxes, scores, labels):
            if score >= threshold:
                xmin, ymin, xmax, ymax = map(int, box)
                cls_name = CLASSES[label] if label < len(CLASSES) else f"CLS_{label}"
                
                cv2.rectangle(img_pred, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(img_pred, f"{cls_name} {score:.2f}", (xmin, max(ymin - 8, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 4. Chuyển đổi và hiển thị lên Tkinter
        # Ảnh GT
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        pil_gt = Image.fromarray(img_gt_rgb)
        self.tk_image_gt = ImageTk.PhotoImage(pil_gt) # Lưu reference
        self.image_label_gt.config(image=self.tk_image_gt, text="")

        # Ảnh Prediction
        img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
        pil_pred = Image.fromarray(img_pred_rgb)
        self.tk_image_pred = ImageTk.PhotoImage(pil_pred) # Lưu reference
        self.image_label_pred.config(image=self.tk_image_pred, text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = CompareViewerApp(root)
    root.mainloop()