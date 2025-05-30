from ultralytics import YOLO


model_path = 'runs/train/my_yolo_training_run/weights/best.pt'
custom_model = YOLO(model_path)

# Perform inference
results = custom_model.predict(source='path/to/an/image.jpg', save=True, conf=0.5)
# 'source' can be an image path, video path, directory, URL, or camera stream
# 'save=True' will save the image with bounding boxes
# 'conf=0.5' sets the confidence threshold for detections

for r in results:
    boxes = r.boxes  # Boxes object for bounding box outputs
    masks = r.masks  # Masks object for segmentation masks outputs
    keypoints = r.keypoints  # Keypoints object for pose outputs
    probs = r.probs  # Probs object for classification outputs
    # r.show()  # display to screen
    # r.save(filename='result.jpg')  # save to disk