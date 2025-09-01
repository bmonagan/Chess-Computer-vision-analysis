
"""
Module for visualizing the confidence distribution of wrong predictions per class using boxplots.
Reads a CSV file with detection results, filters for incorrect predictions, 
and saves a boxplot image.
"""
# Standard Imports
import os
# Third-Party Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

#swited computers so should updated paths if i want to use this again in the future.
# should probably not hardcode paths like this as well or use a env or config file.
CSV_PATH = (
    r'C:\Users\Owner.BLIZZCON\Documents\Programming\Projects'
    r'\Chess Computer vision analysis\my_inference_outputs'
    r'\fine_tuning_20250603\threshold_0.5'
    r'\fine_tuning_test_data_detections_with_labels.csv'
)
df = pd.read_csv(CSV_PATH)


# Convert correct_label to boolean if needed
df['correct_label'] = df['correct_label'].astype(str).map({'True': True, 'False': False})

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

# Ensure the graphs directory exists
GRAPHS_DIR = (
    r'C:\Users\Owner.BLIZZCON\Documents\Programming\Projects'
    r'\Chess Computer vision analysis\my_inference_outputs'
    r'\fine_tuning_20250603\threshold_0.5\graphs'
)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Then use this for saving
plt.savefig(os.path.join(GRAPHS_DIR, 'wrong_confidence_boxplot.png'), dpi=300, bbox_inches='tight')
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
plt.savefig(os.path.join(GRAPHS_DIR, 'wrong_percentage_barplot.png'), dpi=300, bbox_inches='tight')
plt.show()

# Calculate correct predictions per class
correct_df = df[df['correct_label'] == True]
correct_counts = correct_df['class_name'].value_counts().reset_index()
correct_counts.columns = ['class_name', 'correct_count']

# Merge with total counts and compute correct percentage
merged_correct = pd.merge(correct_counts, total_counts, on='class_name', how='left')
correctCount = merged_correct['correct_count']
totalCount = merged_correct['total_count']
merged_correct['correct_pct'] = (correctCount / totalCount) * 100

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
plt.savefig(os.path.join(GRAPHS_DIR,
                         'correct_percentage_barplot.png'),
                           dpi=300, bbox_inches='tight'
)
plt.show()

print("CSV exists:", os.path.isfile(CSV_PATH))
print("DataFrame shape:", df.shape)
print(df.head())
