cd fastercnn-pytorch-training-pipeline
TORCH_HOME="./torch_cache" CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 train.py --data data_configs/fisheye8k_yolo_format.yaml --epochs 80 --model fasterrcnn_resnet50_fpn --name fisheye8k_frcnn_resnet50 --batch 8 --label-type yolo



# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# onnx 1.20.1 requires protobuf>=4.25.1, but you have protobuf 3.20.1 which is incompatible.
# unbabel-comet 2.2.7 requires protobuf<5.0.0,>=4.24.4, but you have protobuf 3.20.1 which is incompatible.
# tensorflow 2.21.0 requires protobuf<8.0.0,>=6.31.1, but you have protobuf 3.20.1 which is incompatible.


TORCH_HOME="./torch_cache" CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 eval.py --data data_configs/fisheye8k_yolo_format.yaml --weights outputs/training/fisheye8k_frcnn_resnet50/best_model.pth --model fasterrcnn_resnet50_fpn 


python eval.py --data data_configs/voc.yaml --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small