import os # For path manipulation, e.g., getting basenames of files
from ultralytics import YOLO
from PIL import Image # Example if you want to do more with r.plot() PIL image
import torch # If you want to use torch for tensor operations or accessing device info

# Load model and define source
model_path = 'runs/detect/my_yolo_training_run7/weights/best.pt'
custom_model = YOLO(model_path)

source_directory = 'testing/images' # image directory

prediction_results_path = custom_model.predict(
    source=source_directory,       # Using the directory as source
    save=True,                     # Save images with detections
    conf=0.5,                      # Confidence threshold
    project='my_inference_outputs',      # Parent directory for these prediction runs
    name='predictions_set1_threshold0.5', # Specific sub-directory for this prediction run
    exist_ok=True,                 # If True, won't increment run number if 'name' exists
                                   # Set to False if you always want a new numbered run
    save_txt=True,                 # Save results as .txt files (YOLO format labels)
    save_conf=True,                # Include confidence scores in --save-txt labels
    save_crop=False,               # Set to True to save cropped images of detections
    line_width=1,                  # Thickness of bounding box lines
    show_labels=True,              # Show labels on bounding boxes
    show_conf=True                 # Show confidence scores on bounding boxes
)

print(f"Prediction outputs (annotated images, text files if save_txt=True) are saved in directories starting from: my_inference_outputs/predictions_set1_threshold0.5/")
if isinstance(prediction_results_path, str): # For single image/video, predict might return path directly
    print(f"Main results saved to: {prediction_results_path}")


# Loop to process results for each source image/frame ---
results_generator = prediction_results_path

for i, r in enumerate(results_generator): 
    original_image_path = r.path  # Path to the original image file
    base_filename = os.path.basename(original_image_path)
    print(f"\n--- Processing results for: {original_image_path} ({i+1} of {len(results_generator) if hasattr(results_generator, '__len__') else 'N/A'}) ---")

        # 3. Accessing bounding box information
    if r.boxes is not None:
        print(f"  Detected {len(r.boxes)} objects:")
        for box_index in range(len(r.boxes)):
            class_id = int(r.boxes.cls[box_index])
            class_name = r.names[class_id]
            confidence = float(r.boxes.conf[box_index])
            xyxy = r.boxes.xyxy[box_index].tolist()
            print(f"    Box {box_index+1}: Class='{class_name}' (ID {class_id}), Confidence={confidence:.2f}, Coordinates (xyxy)={xyxy}")
            detection_data = {
                "image_path": original_image_path,
                "class_name": class_name,
                "confidence": confidence,
                "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]
            }
    else:
        print("  Detected 0 objects.")

    # 4. Accessing masks (if it's a segmentation model and masks are present)
    if r.masks is not None:
        print(f"  Detected {len(r.masks)} masks.")
        # For mask details, you'd access r.masks.data, r.masks.xy, etc.
        # Example:
        # for mask_index, mask_polygon_points in enumerate(r.masks.xy):
        #     print(f"    Mask {mask_index+1} polygon points: {mask_polygon_points[:3]}... (first 3 points)")
        # Segmented objects can also be plotted or saved.
        # r.plot(masks=True) # To get an image with masks visualized (returns a NumPy array)
    else:
        print("  No masks in this result (or not a segmentation model).")

    # Accessing keypoints 
    if r.keypoints is not None:
        print(f"  Detected {len(r.keypoints)} sets of keypoints.")
        # For keypoint details, access r.keypoints.xy, r.keypoints.conf
        # Example:
        # for kp_index, keypoints_for_instance in enumerate(r.keypoints.xy): # Keypoints for one detected instance
        #      print(f"    Keypoints set {kp_index+1}: {keypoints_for_instance[:2]}... (first 2 keypoints x,y)")
    else:
        print("  No keypoints in this result (or not a pose estimation model).")

    # Accessing classification probabilities 
    # For pure object detection, r.probs might be None or less relevant unless it's also doing classification.
    # If your model is a classifier, r.probs (or r.cls for top class, r.names for mapping) would be key.
    if r.probs is not None:
        probs_tensor = r.probs.data if hasattr(r.probs, "data") else r.probs
        if not isinstance(probs_tensor, torch.Tensor):
            probs_tensor = torch.tensor(probs_tensor)
        top5_probs, top5_indices = torch.topk(probs_tensor, 5)
        print("  Top 5 Classification Probabilities (if applicable):")
        for k in range(len(top5_probs)):
            class_id = top5_indices[k].item()
            prob = top5_probs[k].item()
            class_name = r.names[int(class_id)] if int(class_id) < len(r.names) else f"Unknown Class {int(class_id)}"
            print(f"    {k+1}. Class='{class_name}' (ID {class_id}), Probability={prob:.4f}")
    else:
        print("  No classification probabilities (probs) in this result.")

    # Get the plotted image as a NumPy array (BGR format by default)
    # This is the same image that r.show() displays or r.save() saves.
    # annotated_image_numpy = r.plot()
    # You can then use OpenCV (cv2) or Pillow (PIL) to do further processing/saving with this array.
    # Example with PIL:
    # pil_image = Image.fromarray(annotated_image_numpy[..., ::-1]) # Convert BGR to RGB for PIL
    # pil_image.save(os.path.join(custom_save_dir, f"pil_saved_{base_filename}"))

print("\n--- Finished processing all prediction results. ---")

# Example: If you collected detection_data into a list of dictionaries, save to CSV here
# import pandas as pd
# all_detections_list = [] # Assume this was populated in the loop
# if all_detections_list:
#   df = pd.DataFrame(all_detections_list)
#   df.to_csv("all_my_detections.csv", index=False)
#   print("Saved all detection data to all_my_detections.csv")