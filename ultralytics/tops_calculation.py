from ultralytics import YOLO
import torch
import thop
from thop import profile
import cv2
import time
import logging
import os

# Suppress YOLO logging
# logging.getLogger('ultralytics').setLevel(logging.ERROR)

if __name__ == "__main__":
    model_path = "/home/ubuntu/YOLOv8-multi-task/runs/multi/train_finetune_output/weights/best.pt"
    val_path = "/ephemeral/work/voc-adas-yolo/images/val2017"
    # Load the YOLO model
    model = YOLO(model_path , task='multi')
    frame_count = len(os.listdir(val_path))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set up dummy input for FLOPs calculation
    dummy_in = torch.randn(1, 3, 1088, 1088).to(device)
    mac, params = profile(model.model, inputs=(dummy_in,))

    # Compute TOPS (Tera Operations Per Second)
    flop = mac * 2
    tops = (flop) / 1e12
    tops_at_30fps = (30 * tops)

    print(f'MACs: {mac:.2f}')
    print(f'TOPS @ 30FPS: {tops_at_30fps:.2f}')
