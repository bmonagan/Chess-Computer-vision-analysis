import glob
import os
print("Images found:", glob.glob(os.path.join("testing/images", "**", "*.*"), recursive=True))