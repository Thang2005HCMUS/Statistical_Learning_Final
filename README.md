Tải dataset:

```
# có thể dùng tài khoản kaggle token của t nếu lười setup :>
set KAGGLE_API_TOKEN=
python load_dataset.py 
# sau khi chạy tải dataset về thì nó sẽ ở thư mục ./data/Fisheye8K, sau đó chạy chia tập train / valid, bước này training mới cần làm 
python split_data.py
```

Tạo môi trường:

```

pip install -r requirements.txt
pip install -r requirements_blackwell.txt
pip install -U ultralytics
```

training:

```
CUDA_VISIBLE_DEVICES=0 python train_yolo.py

CUDA_VISIBLE_DEVICES=0 python train_rtdetr.py

cd fastercnn-pytorch-training-pipeline
TORCH_HOME="./torch_cache" CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 train.py --data data_configs/fisheye8k_yolo_format.yaml --epochs 80 --model fasterrcnn_resnet50_fpn --name fisheye8k_frcnn_resnet50 --batch 8 --label-type yolo
```

evaluate:
```
CUDA_VISIBLE_DEVICES=0 python eval_yolo.py

CUDA_VISIBLE_DEVICES=0 python eval_rtdetr.py
```

inference, note: change the model in config part

```
CUDA_VISIBLE_DEVICES=0 python infer.py
```


