import sys
import os

project_root = os.path.dirname(__file__) + "/../../"
project_root = os.path.abspath(project_root)
source_root = os.path.dirname(__file__) + "/../"
source_root = os.path.abspath(source_root)

sys.path.insert(0, source_root+"/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
model = YOLO(project_root+'/pretrained/ayolom-voc-1080-20250218.pt', task='multi')

# Train the model
model.train(data=source_root+'/ultralytics/datasets/bdd-multi-voc-finetune.yaml', batch=30, epochs=200, imgsz=(1080,1080), device=0, name='train_finetune_output', val=True, task='multi',classes=[0,1,2,3,4,5,6],combine_class=[],single_cls=False, save_period=5, workers=16, resume=False, lr0=0.003)
