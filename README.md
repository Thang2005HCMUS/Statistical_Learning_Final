# Fisheye8K Object Detection

## 1. Download Dataset

Thiết lập Kaggle API Token:

```bash
export KAGGLE_API_TOKEN=<your_token>
```

Tải dataset:

```bash
python load_dataset.py
```

Sau khi tải xong, dataset sẽ được lưu tại:

```text
./data/Fisheye8K
```

Để tạo tập train/validation (80/20) (chỉ cần thực hiện khi huấn luyện mô hình):

```bash
python split_data.py
```

---

## 2. Environment Setup

Cài đặt các thư viện cần thiết:

```bash
cd fastercnn-pytorch-training-pipeline

pip install -r requirements.txt
pip install -r requirements_blackwell.txt
pip install -U ultralytics
```

---

## 3. Training

### YOLO

```bash
CUDA_VISIBLE_DEVICES=0 python train_yolo.py
```

### RT-DETR

```bash
CUDA_VISIBLE_DEVICES=0 python train_rtdetr.py
```

### Faster R-CNN

```bash
cd fastercnn-pytorch-training-pipeline

TORCH_HOME="./torch_cache" \
CUDA_VISIBLE_DEVICES=3 \
torchrun --nproc_per_node=1 train.py \
    --data data_configs/fisheye8k_yolo_format.yaml \
    --epochs 80 \
    --model fasterrcnn_resnet50_fpn \
    --name fisheye8k_frcnn_resnet50 \
    --batch 8 \
    --label-type yolo
```

---

## 4. Evaluation
Evaluate trên tập test cung cấp của Fisheye8K, đường dẫn `./data/Fisheye8K/test`

### YOLO

```bash
CUDA_VISIBLE_DEVICES=0 python eval_yolo.py
```

### RT-DETR

```bash
CUDA_VISIBLE_DEVICES=0 python eval_rtdetr.py
```

---

## 5. Inference

Trước khi chạy inference, hãy cập nhật đường dẫn model trong phần config trong file `infer.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py
```

---

## Project Structure

```text
.
├── data/
│   ├── Fisheye8K_train_split/ # có sau khi chạy file `split_data.py` để thực hiện training
│   └── Fisheye8K/ # tải xuống bằng kaggle API
├── load_dataset.py
├── split_data.py
├── train_yolo.py
├── train_rtdetr.py
├── eval_yolo.py
├── eval_rtdetr.py
├── infer.py
└── fastercnn-pytorch-training-pipeline/
```
