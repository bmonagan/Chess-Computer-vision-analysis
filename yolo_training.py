from ultralytics import YOLO
import os
import json
from datetime import datetime


def main():
    # yolo model
    # model_name = 'models/yolov8n.pt' 
    # model = YOLO(model_name)
    model_name = "best.pt"
    model = YOLO("runs/detect/my_yolo_training_run_20250603_1929262/weights/best.pt")

    # run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_name = f"my_yolo_training_run_{run_id}"
    run_name = "fine_tune_yolo_run_20250603_1929262"
    training_settings = {
        "model_name": model_name,
        "data": 'Merge_chess.v1i.yolov8/data.yaml',
        "epochs": 50,
        "imgsz": 416,
        "batch": 8,
        "name": run_name,
        "device": "cpu",
        "workers": 1,
        "patience": 30,
        "amp": False,
    }

    results = model.train(
        data=training_settings["data"],
        epochs=training_settings["epochs"],
        imgsz=training_settings["imgsz"],
        batch=training_settings["batch"],
        name=training_settings["name"],
        device=training_settings["device"],
        workers=training_settings["workers"],
        patience=training_settings["patience"],
        amp=training_settings["amp"],
        project=os.path.join("runs", "detect"),  
    )


if __name__ == '__main__':
    main()