from ultralytics import YOLO
import os
import json
from datetime import datetime


def main():
    model_name = 'yolov8n.pt' 
    model = YOLO(model_name)

    # Create a unique directory for each run using timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("runs", "detect", f"my_yolo_training_run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    # Training settings to save
    training_settings = {
        "model_name": model_name,
        "data": 'Merge_chess.v1i.yolov8/data.yaml',
        "epochs": 100,
        "imgsz": 416,
        "batch": 4,
        "name": f"my_yolo_training_run_{run_id}",
        "device": "cpu",
        "workers": 1,
        "patience": 30,
        "amp": False,
    }

    # Save settings as JSON in the unique run directory
    with open(os.path.join(save_dir, "training_settings.json"), "w") as f:
        json.dump(training_settings, f, indent=4)

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
    if results is not None and hasattr(results, "save_dir"):
        print("Training complete. Best model saved at:", results.save_dir)
    else:
        print("Training did not return results as expected. Check for errors above.")


if __name__ == '__main__':
    main()