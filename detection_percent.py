import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


csv_path = os.path.join(os.path.dirname(__file__), r'C:\Users\Owner.BLIZZCON\Documents\Programming\Projects\Chess Computer vision analysis\my_inference_outputs_20250603_1929262\detection_percent.csv')
df = pd.read_csv(csv_path)


df['percent detected'] = df['percent detected'].str.rstrip('%').astype('float')
df['Picture #'] = df['Picture #'].astype(str)


num_pictures = len(df['Picture #'].unique())
figure_width = max(8, min(1.0 * num_pictures, 30))
plt.figure(figsize=(figure_width, 7))

sns.barplot(
    data=df,
    x='Picture #',
    y='percent detected',
    palette='Greens'
)
plt.title('Percentage of Detected Pieces per Picture')
plt.xlabel('Picture Number')
plt.ylabel('Detection Percentage (%)')

if df['percent detected'].max() <= 100:
    plt.ylim(0, 100)
else:
    plt.ylim(0, df['percent detected'].max() + 5) 

plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.savefig('my_inference_outputs_20250603_1929262/detection_percentage_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

