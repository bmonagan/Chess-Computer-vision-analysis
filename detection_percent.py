import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

csv_path = r'C:\Users\Owner.BLIZZCON\Documents\Programming\Projects\Chess Computer vision analysis\my_inference_outputs\fine_tuning_20250603\threshold_0.5\fine_tuning_detection_percent.csv'
df = pd.read_csv(csv_path)

# Convert percent detected to percent
df['percent detected'] = df['percent detected'] * 100

plt.figure(figsize=(max(8, min(1.0 * len(df), 30)), 7))
sns.barplot(
    data=df,
    x='Picture #',
    y='percent detected',
    palette='Greens'
)

# Add average line
avg = df['percent detected'].mean()
plt.axhline(avg, color='red', linestyle='--', label=f'Average: {avg:.1f}%')

plt.title('Detection Percentage per Picture')
plt.xlabel('Picture Number')
plt.ylabel('Detection Percentage (%)')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('my_inference_outputs/fine_tuning_20250603/threshold_0.5/graphs/fine_tuning_detection_percentage_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

