import sys
sys.path.insert(0, "/home/ubuntu/YOLOv8-multi-task/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
model = YOLO('./models/voc/yolov8-bdd-v4-one-dropout-individual-s.yaml', task='multi') # build from YAML and transfer weights

# Train the model
model.train(data='./datasets/bdd-multi-voc-base.yaml', batch=30, epochs=100, imgsz=(1080,1080), device=0, name='train_base_output', val=True, task='multi',classes=[0,1,2,3,4,5,6],combine_class=[],single_cls=False, save_period=5, workers=16, resume=False)
