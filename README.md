# Chess Computer Vision Analysis

This project uses computer vision and deep learning (YOLOv8) to detect and analyze chess pieces on a board from images. It includes scripts for training, inference, and statistical analysis of detection results.

## Features

- YOLOv8-based object detection for chess pieces
- Automated training and evaluation scripts
- CSV export of detection results
- Confidence and accuracy graphing with Seaborn/Matplotlib
- Batch processing and results visualization

## Project Structure
Model Training: yolo_training.py

Interface into model for inferences and save results: interface.py

Analyze result accuracy with graphing: python confidence_graphing.py, python detection_percent.py

Monitor Training Results: tensorboard --logdir runs/detect

## Results
![detection percentage](my_inference_outputs\fine_tuning_20250603\threshold_0.5\graphs\fine_tuning_detection_percentage_barplot.png)
![confidence boxplot](my_inference_outputs\fine_tuning_20250603\threshold_0.5\graphs\wrong_confidence_boxplot.png)



## Conclusions
Current hardware led to limitations in the testing. 

Detection is getting closer but is not in a useable state for my original purpose.

Was trying out UV on this project as well which created some dependencies issues when i accidently pip installed torchvision but did not add it to the UV list. 
In the future i think it will be very useful but some initial growing pains with it.


## Improvements & Future Work

Use a segmentation model to get the exact shapes of the pieces using masks.

Options for hardware limitations include using an online gpu source like google colab or something similar.

Another option would be to just set aside the time for the model to run when im not using the conmputer as runnign the model on gpu created a large amount of lag in all the other processes.



With an accurate enough detection model the plan was to create a pipeline for taking images of a chess board and then getting a computer evaluation of the chess position.