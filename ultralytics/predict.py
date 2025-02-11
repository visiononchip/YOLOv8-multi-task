import sys
sys.path.insert(0, "/home/ubuntu/voc-us-west/YOLOv8-multi-task/ultralytic")

from ultralytics import YOLO


number = 3 #input how many tasks in your work
model = YOLO('/home/ubuntu/voc-us-west/YOLOv8-multi-task/runs/multi/train_demo_output7/weights/best.pt')  # Validate the model
model.predict(source='/home/ubuntu/voc-us-west/train_test/images/val2017', imgsz=(384,672), device=[0],name='predict_demo', save=True, conf=0.25, iou=0.45, show_labels=False, save_txt=True)
