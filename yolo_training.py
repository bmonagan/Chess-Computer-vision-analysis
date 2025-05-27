from ultralytics import YOLO


def main():
    model_name = 'YOLO11m.pt'
    model = YOLO(model_name)
    results = model.train(
        data='data_set\\Merge_chess.v1i.yolov11\\data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='my_yolo_training_run',
        device="0",
        workers=1,
        patience=10,
    )
    if results is not None and hasattr(results, "save_dir"):
        print("Training complete. Best model saved at:", results.save_dir)
    else:
        print("Training did not return results as expected. Check for errors above.")


if __name__ == '__main__':
    main()