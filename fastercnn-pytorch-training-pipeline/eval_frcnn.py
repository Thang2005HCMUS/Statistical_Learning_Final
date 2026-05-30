import os
import glob
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

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
        
        self.image_paths = glob.glob(os.path.join(img_dir, '*.png')) + \
                           glob.glob(os.path.join(img_dir, '*.jpg')) + \
                           glob.glob(os.path.join(img_dir, '*.jpeg'))
                           
        print(f"[*] Đã tìm thấy {len(self.image_paths)} ảnh trong {img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        img_name = os.path.basename(img_path)
        base_name = img_name.rsplit('.', 1)[0]
        label_path = os.path.join(self.label_dir, base_name + '.txt')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, (self.imgsz, self.imgsz))
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        
                        xmin = (cx - w / 2) * self.imgsz
                        ymin = (cy - h / 2) * self.imgsz
                        xmax = (cx + w / 2) * self.imgsz
                        ymax = (cy + h / 2) * self.imgsz
                        
                        boxes.append([xmin, ymin, xmax, ymax])
                        # VÌ TRONG YAML ĐÃ CÓ '__background__' Ở INDEX 0,
                        # NÊN class_id CỦA YOLO (0=bus, 1=bike...) KHI CỘNG 1 SẼ KHỚP CHÍNH XÁC VỚI INDEX TRONG YAML
                        labels.append(int(class_id) + 1) 

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------------------------------------
# 3. CÁC HÀM PHỤ TRỢ TÍNH TOÁN METRICS & VẼ ĐỒ THỊ KIỂU YOLO
# ---------------------------------------------------------
def calculate_confusion_matrix(preds_list, target_list, num_classes, iou_thres=0.45, conf_thres=0.25):
    """Tính toán ma trận nhầm lẫn chuẩn xác theo index lớp từ 1 đến num_classes"""
    # Kích thước matrix: số class thật + 1 hàng/cột cho background hệ thống
    matrix = np.zeros((num_classes + 1, num_classes + 1))
    
    for preds, targets in zip(preds_list, target_list):
        p_boxes = preds['boxes']
        p_labels = preds['labels']
        p_scores = preds['scores']
        
        t_boxes = targets['boxes']
        t_labels = targets['labels']
        
        keep = p_scores >= conf_thres
        p_boxes = p_boxes[keep]
        p_labels = p_labels[keep]
        
        if len(p_boxes) == 0 and len(t_boxes) == 0:
            continue
            
        if len(t_boxes) == 0:
            for pl in p_labels:
                if pl <= num_classes:
                    matrix[pl - 1, num_classes] += 1
            continue
            
        if len(p_boxes) == 0:
            for tl in t_labels:
                if tl <= num_classes:
                    matrix[num_classes, tl - 1] += 1
            continue
            
        iou = box_iou(p_boxes, t_boxes)
        x = torch.where(iou >= iou_thres)
        
        matches = []
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if matches.shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        
        matched_p = set()
        matched_t = set()
        for m in matches:
            p_idx, t_idx = int(m[0]), int(m[1])
            pl = p_labels[p_idx]
            tl = t_labels[t_idx]
            if pl <= num_classes and tl <= num_classes:
                matrix[pl - 1, tl - 1] += 1
            matched_p.add(p_idx)
            matched_t.add(t_idx)
            
        for i in range(len(p_boxes)):
            if i not in matched_p:
                pl = p_labels[i]
                if pl <= num_classes:
                    matrix[pl - 1, num_classes] += 1
                
        for i in range(len(t_boxes)):
            if i not in matched_t:
                tl = t_labels[i]
                if tl <= num_classes:
                    matrix[num_classes, tl - 1] += 1
                
    return matrix

def plot_confusion_matrix(matrix, class_names, output_dir='eval_frcnn'):
    os.makedirs(output_dir, exist_ok=True)
    full_classes = class_names + ['background']
    
    # 1. Vẽ Ma trận gốc
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(full_classes))
    plt.xticks(tick_marks, full_classes, rotation=90)
    plt.yticks(tick_marks, full_classes)
    
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            if val > 0:
                plt.text(j, i, f"{val}", ha="center", va="center",
                         color="white" if matrix[i, j] > thresh else "black")
                         
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()

    # 2. Vẽ Ma trận chuẩn hóa
    plt.figure(figsize=(10, 8))
    with np.errstate(divide='ignore', invalid='ignore'):
        col_sums = matrix.sum(axis=0)
        norm_matrix = np.where(col_sums > 0, matrix / col_sums, 0.0)
    
    plt.imshow(norm_matrix, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix Normalized')
    plt.colorbar()
    plt.xticks(tick_marks, full_classes, rotation=90)
    plt.yticks(tick_marks, full_classes)
    
    for i in range(norm_matrix.shape[0]):
        for j in range(norm_matrix.shape[1]):
            val = norm_matrix[i, j]
            if val > 0.005:
                plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if norm_matrix[i, j] > 0.4 else "black")
                         
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=200)
    plt.close()

def generate_and_plot_curves(preds_list, target_list, class_names, output_dir='eval_frcnn'):
    os.makedirs(output_dir, exist_ok=True)
    conf_values = np.linspace(0, 1, 100)
    num_classes = len(class_names)
    
    p_curve = np.zeros((num_classes, 100))
    r_curve = np.zeros((num_classes, 100))
    f1_curve = np.zeros((num_classes, 100))
    
    for c_idx in range(num_classes):
        target_class_id = c_idx + 1 
        
        total_targets = sum([torch.sum(t['labels'] == target_class_id).item() for t in target_list])
        if total_targets == 0:
            continue
            
        for idx, conf in enumerate(conf_values):
            tp = 0
            fp = 0
            
            for preds, targets in zip(preds_list, target_list):
                p_boxes = preds['boxes']
                p_labels = preds['labels']
                p_scores = preds['scores']
                
                t_boxes = targets['boxes'][targets['labels'] == target_class_id]
                
                mask = (p_labels == target_class_id) & (p_scores >= conf)
                sel_p_boxes = p_boxes[mask]
                
                if len(sel_p_boxes) == 0:
                    continue
                if len(t_boxes) == 0:
                    fp += len(sel_p_boxes)
                    continue
                    
                iou = box_iou(sel_p_boxes, t_boxes)
                max_iou, _ = torch.max(iou, dim=1)
                matched_tp = torch.sum(max_iou >= 0.5).item()
                
                tp += matched_tp
                fp += (len(sel_p_boxes) - matched_tp)
                
            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / total_targets if total_targets > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            p_curve[c_idx, idx] = p
            r_curve[c_idx, idx] = r
            f1_curve[c_idx, idx] = f1

    # Sử dụng vẽ đồ thị dạng an toàn hơn với matplotlib mới
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        plt.plot(conf_values, p_curve[i], label=name, linewidth=1)
    mean_p = np.mean(p_curve, axis=0)
    best_conf_p_idx = np.argmax(mean_p)
    plt.plot(conf_values, mean_p, label=f'all classes {mean_p[best_conf_p_idx]:.2f} at {conf_values[best_conf_p_idx]:.3f}', color='blue', linewidth=3)
    plt.title('Precision-Confidence Curve')
    plt.xlabel('Confidence')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'BoxP_curve.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        plt.plot(conf_values, r_curve[i], label=name, linewidth=1)
    mean_r = np.mean(r_curve, axis=0)
    plt.plot(conf_values, mean_r, label=f'all classes {mean_r[0]:.2f} at 0.000', color='blue', linewidth=3)
    plt.title('Recall-Confidence Curve')
    plt.xlabel('Confidence')
    plt.ylabel('Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'BoxR_curve.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        plt.plot(conf_values, f1_curve[i], label=name, linewidth=1)
    mean_f1 = np.mean(f1_curve, axis=0)
    best_idx = np.argmax(mean_f1)
    plt.plot(conf_values, mean_f1, label=f'all classes {mean_f1[best_idx]:.2f} at {conf_values[best_idx]:.3f}', color='blue', linewidth=3)
    plt.title('F1-Confidence Curve')
    plt.xlabel('Confidence')
    plt.ylabel('F1')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'BoxF1_curve.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        sort_idx = np.argsort(r_curve[i])
        plt.plot(r_curve[i][sort_idx], p_curve[i][sort_idx], label=f'{name} {np.trapz(p_curve[i][sort_idx], r_curve[i][sort_idx]):.3f}', linewidth=1)
    
    sort_mean_idx = np.argsort(mean_r)
    mean_ap_val = np.trapz(mean_p[sort_mean_idx], mean_r[sort_mean_idx])
    plt.plot(mean_r[sort_mean_idx], mean_p[sort_mean_idx], label=f'all classes {mean_ap_val:.3f} mAP@0.5', color='blue', linewidth=3)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'BoxPR_curve.png'), dpi=200)
    plt.close()

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
    parser.add_argument('--project', default='eval_frcnn', help='Folder name to save results')
    args = vars(parser.parse_args())

    device = torch.device(args['device'] if args['device'] else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(f"[*] Đang chạy trên thiết bị: {device}")

    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    try:
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    except KeyError:
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
        
    CLASSES = data_configs['CLASSES'] # Nhận mảng: ['__background__', 'bus', 'bike', 'car', 'pedestrian', 'truck']
    
    # TRÍCH XUẤT CHÍNH XÁC DANH SÁCH CLASS THẬT (BỎ QUA PHẦN TỬ ĐẦU TIÊN)
    real_class_names = [c for c in CLASSES if c != '__background__']
    num_real_classes = len(real_class_names)

    print("[*] Đang khởi tạo Dataset (Hỗ trợ YOLO txt)...")
    valid_dataset = YoloFormatDataset(VALID_DIR_IMAGES, VALID_DIR_LABELS, CLASSES, imgsz=args['imgsz'])
    valid_loader = DataLoader(valid_dataset, batch_size=args['batch'], shuffle=False, num_workers=2, collate_fn=collate_fn)

    print("[*] Đang khởi tạo mô hình Faster R-CNN ResNet50 FPN và nạp trọng số...")
    # Faster R-CNN cần số lượng class tổng bao gồm cả background hệ thống nội bộ
    # Vì file YAML đặt NC: 6 (đã bao gồm background gộp sẵn), ta truyền trực tiếp vào
    model = fasterrcnn_resnet50_fpn.create_model(num_classes=data_configs['NC'], coco_model=False)

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

    print("[*] Đang tính toán mAP từ TorchMetrics...")
    metric.update(preds_list, target_list)
    stats = metric.compute()

    # Tính toán Confusion Matrix theo đúng số lượng lớp thực tế
    cm_matrix = calculate_confusion_matrix(preds_list, target_list, num_classes=num_real_classes)
    
    p_per_class = []
    r_per_class = []
    f1_per_class = []
    for i in range(num_real_classes):
        tp = cm_matrix[i, i]
        fp = sum(cm_matrix[i, :]) - tp
        fn = sum(cm_matrix[:, i]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        p_per_class.append(precision)
        r_per_class.append(recall)
        f1_per_class.append(f1)
        
    mean_precision = sum(p_per_class) / len(p_per_class)
    mean_recall = sum(r_per_class) / len(r_per_class)
    mean_f1_score = sum(f1_per_class) / len(f1_per_class)

    map50_95 = stats['map'].item()
    map50 = stats['map_50'].item()
    map75 = stats['map_75'].item()

    print("\n--- KẾT QUẢ ĐÁNH GIÁ (Định dạng YOLO chuẩn) ---")
    print(f"mAP (50-95): {map50_95:.4f}")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP75: {map75:.4f}")
    print(f"Mean F1 Score: {mean_f1_score:.4f}")
    
    print("\nClass indices with average precision:", list(range(num_real_classes)))
    print("Average precision per class:", [round(x.item(), 4) for x in stats['map_per_class']])
    print("Average precision at IoU=0.50:", map50)
    print("F1 score:", [round(x, 4) for x in f1_per_class])
    
    print(f"\nMean average precision: {map50_95:.4f}")
    print(f"Mean average precision at IoU=0.50: {map50:.4f}")
    print(f"Mean average precision at IoU=0.75: {map75:.4f}")
    print(f"Mean precision: {mean_precision:.4f}")
    print(f"Mean recall: {mean_recall:.4f}")
    print("Precision:", [round(x, 4) for x in p_per_class])
    print("Recall:", [round(x, 4) for x in r_per_class])

    print(f"\n[*] Đang tạo biểu đồ và ma trận nhầm lẫn tại thư mục '{args['project']}'...")
    plot_confusion_matrix(cm_matrix, real_class_names, output_dir=args['project'])
    generate_and_plot_curves(preds_list, target_list, real_class_names, output_dir=args['project'])
    print("[*] Hoàn thành! Ma trận nhầm lẫn đã được căn chỉnh trục lớp chính xác.")