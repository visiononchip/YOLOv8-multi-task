import sys
sys.path.insert(0, "/home/ubuntu/YOLOv8-multi-task/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
model = YOLO('../base_model_720.pt', task='multi')

# Train the model
model.train(data='./datasets/bdd-multi-voc-finetune.yaml', batch=96, epochs=100, imgsz=(640,640), device=0, name='train_finetune_output', val=True, task='multi',classes=[0,1,2,3,4,5,6],combine_class=[],single_cls=False, save_period=5, workers=16, resume=False)
