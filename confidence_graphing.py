import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set_theme(style="darkgrid")


csv_path = os.path.join(os.path.dirname(__file__), r'C:\Users\Owner.BLIZZCON\Documents\Programming\Projects\Chess Computer vision analysis\my_inference_outputs_20250603_1929262\test_data_detections_with_labels.csv')
df = pd.read_csv(csv_path)


# Convert correct_label to boolean if needed
df['correct_label'] = df['correct_label'].map({'True': True, 'False': False})

# Filter for wrong predictions
wrong_df = df[df['correct_label'] == False].copy()
wrong_df['confidence_pct'] = wrong_df['confidence'] * 100  # Convert to percent

plt.figure(figsize=(14, 7))
sns.boxplot(
    data=wrong_df,
    x='class_name',
    y='confidence_pct',
    palette='Reds'
)
plt.title('Confidence Distribution for Wrong Predictions per Class')
plt.xlabel('Class Name')
plt.ylabel('Confidence (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('my_inference_outputs_20250603_1929262/wrong_confidence_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate total predictions per class
total_counts = df['class_name'].value_counts().reset_index()
total_counts.columns = ['class_name', 'total_count']

# Calculate wrong predictions per class
wrong_counts = wrong_df['class_name'].value_counts().reset_index()
wrong_counts.columns = ['class_name', 'wrong_count']

# Merge and compute percentage
merged = pd.merge(wrong_counts, total_counts, on='class_name', how='left')
merged['wrong_pct'] = (merged['wrong_count'] / merged['total_count']) * 100

plt.figure(figsize=(14, 7))
sns.barplot(
    data=merged,
    x='class_name',
    y='wrong_pct',
    palette='Reds'
)
plt.title('Percentage of Wrong Predictions per Class')
plt.xlabel('Class Name')
plt.ylabel('Wrong Prediction Percentage (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('my_inference_outputs_20250603_1929262/wrong_percentage_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate correct predictions per class
correct_df = df[df['correct_label'] == True]
correct_counts = correct_df['class_name'].value_counts().reset_index()
correct_counts.columns = ['class_name', 'correct_count']

# Merge with total counts and compute correct percentage
merged_correct = pd.merge(correct_counts, total_counts, on='class_name', how='left')
merged_correct['correct_pct'] = (merged_correct['correct_count'] / merged_correct['total_count']) * 100

plt.figure(figsize=(14, 7))
sns.barplot(
    data=merged_correct,
    x='class_name',
    y='correct_pct',
    palette='Greens'
)
plt.title('Percentage of Correct Predictions per Class')
plt.xlabel('Class Name')
plt.ylabel('Correct Prediction Percentage (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('my_inference_outputs_20250603_1929262/correct_percentage_barplot.png', dpi=300, bbox_inches='tight')
plt.show()