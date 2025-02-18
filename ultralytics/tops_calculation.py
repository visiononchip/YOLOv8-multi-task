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
    model_path = "best.pt"
    val_path = "PATH to VAL2017"
    # Load the YOLO model
    model = YOLO(model_path , task='multi')
    frame_count = len(os.listdir(val_path))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set up dummy input for FLOPs calculation
    dummy_in = torch.randn(1, 3, 1088, 1088).to(device)
    flops, params = profile(model.model, inputs=(dummy_in,))

    results = model.val(data="DATASET YAML", imgsz=1088, device=[0], conf=0.25, iou=0.45, half=False, show_labels=False, save=False, show=False, save_txt=True)

    inference_time = results[0].speed['preprocess'] + results[0].speed['inference'] + results[0].speed['loss'] + results[0].speed['postprocess']

    total_inference_ms = (frame_count * inference_time) / 1000 # convert ms to seconds and multiply with total number of frames

    overall_fps = frame_count / total_inference_ms if total_inference_ms > 0 else 0

    # Compute TOPS (Tera Operations Per Second)
    tops = (frame_count * flops / total_inference_ms) / 1e12

    tops_at_30fps = tops * 30 / overall_fps

    print(f'TOPS @ 30FPS: {tops_at_30fps:.2f}')
