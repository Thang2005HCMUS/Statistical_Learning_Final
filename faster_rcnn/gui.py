import os
import glob
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Danh sách class của bạn (Index 0 là background)
CLASSES = ['background', 'bus', 'bike', 'car', 'pedestrian', 'truck']
IMGSZ = 640

class YoloDebugViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Bounding Box Debugger")
        self.root.geometry("1000x700")
        
        self.img_dir = ""
        self.label_dir = ""
        self.image_paths = []
        
        self.setup_ui()

    def setup_ui(self):
        # Frame Trái (Chứa nút bấm và danh sách)
        left_frame = tk.Frame(self.root, width=300, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Các nút chọn thư mục
        tk.Button(left_frame, text="1. Chọn thư mục Ảnh", command=self.browse_img_dir, width=25).pack(pady=5)
        self.lbl_img_dir = tk.Label(left_frame, text="Chưa chọn...", fg="blue", wraplength=250)
        self.lbl_img_dir.pack(pady=5)

        tk.Button(left_frame, text="2. Chọn thư mục Nhãn (.txt)", command=self.browse_label_dir, width=25).pack(pady=5)
        self.lbl_label_dir = tk.Label(left_frame, text="Chưa chọn...", fg="blue", wraplength=250)
        self.lbl_label_dir.pack(pady=5)

        tk.Label(left_frame, text="Danh sách ảnh:").pack(anchor=tk.W, pady=(10, 0))
        
        # Listbox và Scrollbar
        list_frame = tk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.scrollbar = tk.Scrollbar(list_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(list_frame, yscrollcommand=self.scrollbar.set, width=35)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.listbox.yview)
        
        # Bắt sự kiện click vào listbox
        self.listbox.bind('<<ListboxSelect>>', self.on_select_image)

        # Frame Phải (Chứa Ảnh)
        self.right_frame = tk.Frame(self.root, bg="gray")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_label = tk.Label(self.right_frame, text="Ảnh sẽ hiển thị ở đây", bg="gray", fg="white")
        self.image_label.pack(expand=True)

    def browse_img_dir(self):
        folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        if folder:
            self.img_dir = folder
            self.lbl_img_dir.config(text=folder)
            self.load_image_list()
            
            # Tự động đoán thư mục nhãn (nếu có thư mục 'labels' ngang hàng với 'images')
            if not self.label_dir:
                guess_label_dir = folder.replace('images', 'labels')
                if os.path.exists(guess_label_dir) and guess_label_dir != folder:
                    self.label_dir = guess_label_dir
                    self.lbl_label_dir.config(text=self.label_dir)
                else:
                    self.label_dir = folder # Mặc định nhãn nằm chung thư mục ảnh
                    self.lbl_label_dir.config(text=self.label_dir)

    def browse_label_dir(self):
        folder = filedialog.askdirectory(title="Chọn thư mục chứa nhãn (.txt)")
        if folder:
            self.label_dir = folder
            self.lbl_label_dir.config(text=folder)

    def load_image_list(self):
        self.listbox.delete(0, tk.END)
        # Quét các định dạng ảnh phổ biến
        self.image_paths = glob.glob(os.path.join(self.img_dir, '*.png')) + \
                           glob.glob(os.path.join(self.img_dir, '*.jpg')) + \
                           glob.glob(os.path.join(self.img_dir, '*.jpeg'))
                           
        if not self.image_paths:
            messagebox.showwarning("Trống", "Không tìm thấy ảnh (.png, .jpg) trong thư mục này!")
            return
            
        for path in self.image_paths:
            name = os.path.basename(path)
            self.listbox.insert(tk.END, name)

    def on_select_image(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        img_path = self.image_paths[index]
        
        img_name = os.path.basename(img_path)
        base_name = img_name.rsplit('.', 1)[0]
        label_path = os.path.join(self.label_dir, base_name + '.txt')

        self.display_image(img_path, label_path)

    def display_image(self, img_path, label_path):
        # 1. Đọc và resize ảnh (giống hệt Data Loader)
        image = cv2.imread(img_path)
        if image is None:
            return
            
        image = cv2.resize(image, (IMGSZ, IMGSZ))
        
        # 2. Đọc file txt và vẽ Bounding Box
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        
                        # Tính tọa độ
                        xmin = int((cx - w / 2) * IMGSZ)
                        ymin = int((cy - h / 2) * IMGSZ)
                        xmax = int((cx + w / 2) * IMGSZ)
                        ymax = int((cy + h / 2) * IMGSZ)
                        
                        try:
                            class_name = CLASSES[int(class_id) + 1]
                        except IndexError:
                            class_name = f"Class {int(class_id)}"
                        
                        # Vẽ khung xanh lá, text đỏ
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(image, class_name, (xmin, max(ymin - 5, 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # 3. Chuyển đổi màu BGR (OpenCV) sang RGB (Tkinter/Pillow)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 4. Chuyển thành ImageTk và hiển thị
        pil_image = Image.fromarray(image)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        self.image_label.config(image=self.tk_image, text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloDebugViewer(root)
    root.mainloop()