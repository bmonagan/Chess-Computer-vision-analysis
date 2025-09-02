"""
Module for visualizing detection percentage per picture using barplots.
Reads a CSV file with detection percentages and saves a barplot image with an average line.
"""
# Third-Party Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config

CSV_PATH = config.DETECTION_CSV_PATH
df = pd.read_csv(CSV_PATH)

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
plt.savefig(config.DETECTION_OUTPUT, dpi=300, bbox_inches='tight')
plt.show()
