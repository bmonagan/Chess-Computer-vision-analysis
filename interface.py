"""
Interface for running YOLO model inference on chessboard images.
Handles model loading, image directory selection, and project directory setup for inference outputs.
"""
# Standard Imports
import os
# Library Imports
from ultralytics import YOLO
import torch
import pandas as pd  # Add this import
# Project Imports
import config

# Load model and define source
MODEL_PATH = config.MODEL_PATH
custom_model = YOLO(MODEL_PATH)
choices = {
    "easy": "testing/images/1_easy",
    "medium": "testing/images/2_medium",
    "hard": "testing/images/3_hard",
    "unrealistic": "testing/images/4_unrealistic",
    "testing" : "testing/images/from_test"
}
SOURCE_DIRECTORY = choices["testing"] # image directory

# # Single Image Path
# SOURCE_DIRECTORY= 'testing/images/qg_closeup.jpg'


CONFIDENCE_THRESHOLD = 0.5

# # ----Dynamic Project Directory Creation-----
# # Extract the folder name from the MODEL_PATH
# folder_name = os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH)))
# # Find all number groups in the folder name
# number_sets = re.findall(r'\d+', folder_name)
# # Get the last two number sets (if available)
# if len(number_sets) >= 2:
#     suffix = f"{number_sets[-2]}_{number_sets[-1]}"
# else:
#     suffix = "_".join(number_sets)

# PROJECT_DIR = f"my_inference_outputs_{suffix}"

# Non-Dynamic Choice
PROJECT_DIR = "my_inference_outputs/fine_tuning_20250603"

prediction_results_path = custom_model.predict(
    source=SOURCE_DIRECTORY,       # Using the directory as source
    save=True,                     # Save images with detections
    conf=CONFIDENCE_THRESHOLD,                  # Confidence threshold
    project=PROJECT_DIR,  # Use the dynamically generated project directory
    name=f'threshold_{CONFIDENCE_THRESHOLD}', # Specific sub-directory for this prediction run
    exist_ok=True,                 # If True, won't increment run number if 'name' exists
    save_txt=True,                 # Save results as .txt files (YOLO format labels)
    save_conf=True,                # Include confidence scores in --save-txt labels
    save_crop=False,               # Set to True to save cropped images of detections
    line_width=None,                  # Thickness of bounding box lines
    show_labels=True,              # Show labels on bounding boxes
    show_conf=True                 # Show confidence scores on bounding boxes
)

print(
    "Prediction outputs (annotated images, text files if save_txt=True) " \
    "are saved in directories starting from: " \
    f"my_inference_outputs/predictions_set1_threshold{CONFIDENCE_THRESHOLD}/"
)
# For single image/video, predict might return path directly
if isinstance(prediction_results_path, str):
    print(f"Main results saved to: {prediction_results_path}")


# Loop to process results for each source image/frame ---
results_generator = prediction_results_path

all_detections_list = []  # Collect all detections here

for i, r in enumerate(results_generator):
    original_image_path = r.path
    base_filename = os.path.basename(original_image_path)
    print(f"\n--- Processing results for: {original_image_path} ({i+1} of "
          f"{len(results_generator) if hasattr(results_generator, '__len__') else 'N/A'}) ---"
    )

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
            IMAGE_PATH_SHORT = f"{parent_folder}/{file_name}"
            detection_data = {
                "image_path": IMAGE_PATH_SHORT,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
                "correct_label": "True"  
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

    # Accessing classification probabilities
    if r.probs is not None:
        probs_tensor = r.probs.data if hasattr(r.probs, "data") else r.probs
        if not isinstance(probs_tensor, torch.Tensor):
            probs_tensor = torch.tensor(probs_tensor)
        top5_probs, top5_indices = torch.topk(probs_tensor, 5)
        print("  Top 5 Classification Probabilities (if applicable):")
        for k, value in enumerate(top5_probs):
            class_id = top5_indices[k].item()
            prob = top5_probs[k].item()
            if prob > 0.5:
                class_name = r.names[int(class_id)] \
                    if int(class_id) < len(r.names) else f"Unknown Class {int(class_id)}"
                print(f"    {k+1}. Class='{class_name}' (ID {class_id}), Probability={prob:.4f}")
    else:
        print("  No classification probabilities (probs) in this result.")

print("\n--- Finished processing all prediction results. ---")

# Save all detections to CSV with labels
CSV_PATH = "fine_tuning_test_data_detections_with_labels.csv"
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
    file_exists = os.path.isfile(CSV_PATH)
    df.to_csv(
        CSV_PATH,
        mode='a' if file_exists else 'w',
        header=not file_exists,
        index=False
    )
    print(f"Saved all detection data with labels to {CSV_PATH} (appended if file existed).")
