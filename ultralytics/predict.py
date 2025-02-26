import sys
import os

project_root = os.path.dirname(__file__) + "/../../"
project_root = os.path.abspath(project_root)
source_root = os.path.dirname(__file__) + "/../"
source_root = os.path.abspath(source_root)

sys.path.insert(0, source_root+"/ultralytics")

from ultralytics import YOLO


number = 3 #input how many tasks in your work
model = YOLO(project_root+'/pretrained/ayolom-voc-1080-fine-20250218.tflite', task='multi')  # Validate the model
model.predict(source=project_root+'/dataset/voc-adas-yolo/images/val2017', imgsz=1088, device=[0],name='predict_demo', save=True, conf=0.25, iou=0.45, show_labels=False, save_txt=True)
