from ultralytics import YOLO
import os
import json


def main():
    model_name = 'yolov8n.pt' #Nano version for faster training
    model = YOLO(model_name)
    
    save_dir = os.path.join("runs", "detect", "my_yolo_training_run")
    os.makedirs(save_dir, exist_ok=True)

    # Training settings to save
    training_settings = {
        "model_name": model_name,
        "data": 'data_set/Merge_chess/data.yaml',
        "epochs": 3,
        "imgsz": 416,
        "batch": 1,
        "name": 'my_yolo_training_run',
        "device": "cpu",
        "workers": 1,
        "patience": 20,
        "amp": False,
    }
    # Save settings as JSON
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
    )
    if results is not None and hasattr(results, "save_dir"):
        print("Training complete. Best model saved at:", results.save_dir)
    else:
        print("Training did not return results as expected. Check for errors above.")


if __name__ == '__main__':
    main()