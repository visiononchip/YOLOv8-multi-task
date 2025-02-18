from ultralytics import YOLO

if __name__ == "__main__":

    # Set the parameters
    half_ = False
    int8_ = True
    if half_:
        print("FP16")
    elif int8_:
        print("INT8")
    else:
        print("Full")

    # # # # Load the model
    #model = YOLO("/home/ubuntu/YOLOv8-multi-task/best.pt", task='multi')
    model = YOLO("/home/ubuntu/YOLOv8-multi-task/epoch40.pt", task='multi')

    # Export the model to engine format
    model.export(format="engine", workspace=15000, imgsz=1080, int8=int8_, half=half_, device=0, dynamic=False, nms=False, 
                 data="/home/ubuntu/YOLOv8-multi-task/ultralytics/datasets/bdd-multi-voc-finetune.yaml", batch=1, simplify=True)  # creates 'engine file at same location as pt file.'
                 #data="/home/ubuntu/YOLOv8-multi-task/ultralytics/datasets/bdd-multi-voc-finetune-combined.yaml", batch=1, simplify=True)  # creates 'engine file at same location as pt file.'

    # Load the exported Engine file model
    #trt_model = YOLO("/home/ubuntu/YOLOv8-multi-task/best.engine", task='multi')
    trt_model = YOLO("/home/ubuntu/YOLOv8-multi-task/epoch40.engine", task='multi')
    # Run validation
    results = trt_model.val(data="/home/ubuntu/YOLOv8-multi-task/ultralytics/datasets/bdd-multi-voc-finetune.yaml", imgsz=1080, int8=int8_,half=half_, device=0)
    print(results)