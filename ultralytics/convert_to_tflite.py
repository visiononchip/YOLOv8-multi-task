from ultralytics import YOLO


if __name__ == "__main__":

    half_ = False
    int8_ = True
    if half_:
        print("FP16")
    elif int8_:
        print("INT8")
    else:
        print("Full")

    # Load the YOLO11 model
    model = YOLO("/home/ubuntu/YOLOv8-multi-task/best.pt", task='multi')

    # Export the model to tflite format
    model.export(format="tflite",int8=int8_, half=half_, device=0)

    # Load the exported tflite model
    tflite_model = YOLO("/home/ubuntu/YOLOv8-multi-task/best_saved_model/best_int8.tflite", task='multi')

    # Run inference
    results = tflite_model.val(data="/home/ubuntu/YOLOv8-multi-task/ultralytics/datasets/bdd-multi-voc-finetune.yaml",imgsz=1088, int8=int8_,half=half_, device=0)
