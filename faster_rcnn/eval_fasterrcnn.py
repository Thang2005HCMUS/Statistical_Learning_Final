import os
import glob
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Import trực tiếp module chứa hàm create_model của bạn
from models import fasterrcnn_resnet50_fpn

# ---------------------------------------------------------
# 1. TẠO CUSTOM DATASET ĐỂ ĐỌC TRỰC TIẾP FILE .TXT (YOLO)
# ---------------------------------------------------------
class YoloFormatDataset(Dataset):
    def __init__(self, img_dir, label_dir, classes, imgsz=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.classes = classes
        self.imgsz = imgsz
        
        # Lấy danh sách tất cả file ảnh
        self.image_paths = glob.glob(os.path.join(img_dir, '*.png')) + \
                           glob.glob(os.path.join(img_dir, '*.jpg')) + \
                           glob.glob(os.path.join(img_dir, '*.jpeg'))
                           
        print(f"[*] Đã tìm thấy {len(self.image_paths)} ảnh trong {img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Tìm file .txt tương ứng
        img_name = os.path.basename(img_path)
        base_name = img_name.rsplit('.', 1)[0]
        label_path = os.path.join(self.label_dir, base_name + '.txt')

        # Đọc ảnh và chuyển sang RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize ảnh
        image = cv2.resize(image, (self.imgsz, self.imgsz))
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1) # Chuyển thành dạng [C, H, W]

        boxes = []
        labels = []

        # Đọc file label nếu tồn tại
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        
                        # Convert YOLO format (tỷ lệ) sang VOC format (pixel tuyệt đối)
                        xmin = (cx - w / 2) * self.imgsz
                        ymin = (cy - h / 2) * self.imgsz
                        xmax = (cx + w / 2) * self.imgsz
                        ymax = (cy + h / 2) * self.imgsz
                        
                        boxes.append([xmin, ymin, xmax, ymax])
                        
                        # Faster RCNN quy ước class 0 là background, nên class thực tế bắt đầu từ 1
                        labels.append(int(class_id) + 1) 

        # Xử lý trường hợp ảnh không có nhãn (hoặc file txt trống)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

def collate_fn(batch):
    """Hàm gom các ảnh có số lượng object khác nhau vào chung một batch"""
    return tuple(zip(*batch))

# ---------------------------------------------------------
# 2. HÀM ĐÁNH GIÁ CHÍNH
# ---------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data config yaml file')
    parser.add_argument('-mw', '--weights', required=True, help='Path to trained checkpoint')
    parser.add_argument('-ims', '--imgsz', default=640, type=int, help='Image size')
    parser.add_argument('-b', '--batch', default=4, type=int, help='Batch size')
    parser.add_argument('-d', '--device', default=None, help='Computation device')
    args = vars(parser.parse_args())

    # Thiết lập device
    device = torch.device(args['device'] if args['device'] else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(f"[*] Đang chạy trên thiết bị: {device}")

    # Đọc cấu hình YAML
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    try:
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    except KeyError:
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
        
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']

    # Khởi tạo DataLoader
    print("[*] Đang khởi tạo Dataset (Hỗ trợ YOLO txt)...")
    valid_dataset = YoloFormatDataset(VALID_DIR_IMAGES, VALID_DIR_LABELS, CLASSES, imgsz=args['imgsz'])
    valid_loader = DataLoader(valid_dataset, batch_size=args['batch'], shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Khởi tạo Mô hình trực tiếp
    print("[*] Đang khởi tạo mô hình Faster R-CNN ResNet50 FPN và nạp trọng số...")
    model = fasterrcnn_resnet50_fpn.create_model(num_classes=NUM_CLASSES, coco_model=False)

    # Đọc Weights và tự động sửa lỗi chữ "module."
    checkpoint = torch.load(args['weights'], map_location=device)
    state_dict = checkpoint['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Bắt đầu đánh giá
    print("[*] Bắt đầu quá trình đánh giá (Evaluation)...")
    metric = MeanAveragePrecision(class_metrics=True)
    
    target_list = []
    preds_list = []

    with torch.inference_mode():
        for images, targets in tqdm(valid_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i in range(len(images)):
                true_dict = {
                    'boxes': targets[i]['boxes'].detach().cpu(),
                    'labels': targets[i]['labels'].detach().cpu()
                }
                preds_dict = {
                    'boxes': outputs[i]['boxes'].detach().cpu(),
                    'scores': outputs[i]['scores'].detach().cpu(),
                    'labels': outputs[i]['labels'].detach().cpu()
                }
                preds_list.append(preds_dict)
                target_list.append(true_dict)

    # Tính toán kết quả
    print("[*] Đang tính toán mAP...")
    metric.update(preds_list, target_list)
    stats = metric.compute()

    # In kết quả định dạng đẹp
    print('\n' + '='*50)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (mAP & mAR)")
    print('='*50)
    
    num_hyphens = 70
    print('-'*num_hyphens)
    print(f"| {'Class':<20} | {'AP (IoU 0.5:0.95)':<20} | {'AR (MaxDets=100)':<20} |")
    print('-'*num_hyphens)
    
    # In cho từng class (bỏ qua background là index 0)
    # In cho từng class (bỏ qua background là index 0)
    for i in range(len(CLASSES) - 1):
        class_name = CLASSES[i+1]
        ap = stats['map_per_class'][i].item()
        ar = stats['mar_100_per_class'][i].item()
        print(f"| {class_name:<20} | {ap:<20.3f} | {ar:<20.3f} |")
        
    print('-'*num_hyphens)
    print(f"| {'Trung bình (Average)':<20} | {stats['map'].item():<20.3f} | {stats['mar_100'].item():<20.3f} |")