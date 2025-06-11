import os
from ultralytics import YOLO
from PIL import Image
import torch
import pandas as pd  # Add this import
import re

# Load model and define source
model_path = r"runs\detect\fine_tune_yolo_run_20250603_1929262\weights\best.pt"
custom_model = YOLO(model_path)
choices = {
    "easy": "testing/images/1_easy",
    "medium": "testing/images/2_medium",
    "hard": "testing/images/3_hard",
    "unrealistic": "testing/images/4_unrealistic",
    "testing" : "testing/images/from_test"
}
source_directory = choices["testing"] # image directory

# # Single Image Path
# source_directory = 'testing/images/qg_closeup.jpg'


confidence_threshold = 0.5

# # Extract the folder name from the model_path
# folder_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
# # Find all number groups in the folder name
# number_sets = re.findall(r'\d+', folder_name)
# # Get the last two number sets (if available)
# if len(number_sets) >= 2:
#     suffix = f"{number_sets[-2]}_{number_sets[-1]}"
# else:
#     suffix = "_".join(number_sets)

# project_dir = f"my_inference_outputs_{suffix}"

# Non-Dynamic Choice
project_dir = "my_inference_outputs/fine_tuning_20250603"  # Static directory for this example

prediction_results_path = custom_model.predict(
    source=source_directory,       # Using the directory as source
    save=True,                     # Save images with detections
    conf=confidence_threshold,                  # Confidence threshold
    project=project_dir,  # Use the dynamically generated project directory
    name=f'threshold_{confidence_threshold}', # Specific sub-directory for this prediction run
    exist_ok=True,                 # If True, won't increment run number if 'name' exists
    save_txt=True,                 # Save results as .txt files (YOLO format labels)
    save_conf=True,                # Include confidence scores in --save-txt labels
    save_crop=False,               # Set to True to save cropped images of detections
    line_width=None,                  # Thickness of bounding box lines
    show_labels=True,              # Show labels on bounding boxes
    show_conf=True                 # Show confidence scores on bounding boxes
)

print(f"Prediction outputs (annotated images, text files if save_txt=True) are saved in directories starting from: my_inference_outputs/predictions_set1_threshold{confidence_threshold}/")
if isinstance(prediction_results_path, str): # For single image/video, predict might return path directly
    print(f"Main results saved to: {prediction_results_path}")


# Loop to process results for each source image/frame ---
results_generator = prediction_results_path

all_detections_list = []  # Collect all detections here

for i, r in enumerate(results_generator): 
    original_image_path = r.path
    base_filename = os.path.basename(original_image_path)
    print(f"\n--- Processing results for: {original_image_path} ({i+1} of {len(results_generator) if hasattr(results_generator, '__len__') else 'N/A'}) ---")

    if r.boxes is not None:
        print(f"  Detected {len(r.boxes)} objects:")
        for box_index in range(len(r.boxes)):
            class_id = int(r.boxes.cls[box_index])
            class_name = r.names[class_id]
            confidence = float(r.boxes.conf[box_index])
            xyxy = r.boxes.xyxy[box_index].tolist()
            # Get just the folder and file name
            parent_folder = os.path.basename(os.path.dirname(original_image_path))
            file_name = os.path.basename(original_image_path)
            image_path_short = f"{parent_folder}/{file_name}"
            detection_data = {
                "image_path": image_path_short,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
                "correct_label": "True"  # Example of a user-defined label
            }
            all_detections_list.append(detection_data)  # Add to list
    else:
        print("  Detected 0 objects.")

    # Add a buffer row after each image's detections
    all_detections_list.append({
        "image_path": "---",
        "class_id": "",
        "class_name": "",
        "confidence": "",
        "x1": "", "y1": "", "x2": "", "y2": "",
        "correct_label": ""
    })

    # TODO
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
    # TODO
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

    # TODO ADDITIONAL
    # Get the plotted image as a NumPy array (BGR format by default)
    # This is the same image that r.show() displays or r.save() saves.
    # annotated_image_numpy = r.plot()
    # You can then use OpenCV (cv2) or Pillow (PIL) to do further processing/saving with this array.
    # Example with PIL:
    # pil_image = Image.fromarray(annotated_image_numpy[..., ::-1]) # Convert BGR to RGB for PIL
    # pil_image.save(os.path.join(custom_save_dir, f"pil_saved_{base_filename}"))

print("\n--- Finished processing all prediction results. ---")

# Save all detections to CSV with labels
csv_path = "fine_tuning_test_data_detections_with_labels.csv"
columns = [
    "image_path",
    "class_id",
    "class_name",
    "confidence",
    "x1", "y1", "x2", "y2", #bounding box coordinates
    "correct_label"
]
if all_detections_list:
    df = pd.DataFrame(all_detections_list, columns=columns)
    file_exists = os.path.isfile(csv_path)
    df.to_csv(
        csv_path,
        mode='a' if file_exists else 'w',
        header=not file_exists,
        index=False
    )
    print(f"Saved all detection data with labels to {csv_path} (appended if file existed).")