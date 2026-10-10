import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# 1. Setup paths to read your collected CSV file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "B.csv"

if not CSV_PATH.exists():
    print(f"❌ Could not find the file at: {CSV_PATH}")
    exit()

# 2. Read the CSV data using pandas
df = pd.read_csv(CSV_PATH)

if df.empty:
    print("❌ The CSV file is empty!")
    exit()

print(f"📊 Successfully loaded {len(df)} saved rows of data for letter '{df['label'].iloc[0]}'.")
print("Press any key on the visual window to cycle through your saved shapes. Press ESC to close.")

# MediaPipe Hand Connection lines map to draw a full skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# 3. Loop through your saved entries row by row
for index, row in df.iterrows():
    # Create a clean, blank black image window (500x500 pixels)
    blank_window = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Extract the 21 coordinate pairs back from the columns
    points = []
    for i in range(21):
        # We multiply by 500 to scale the 0.0-1.0 percentages back to pixel locations
        x = int(row[f'x{i}'] * 500)
        y = int(row[f'y{i}'] * 500)
        points.append((x, y))

    # Draw the skeleton connection lines
    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(blank_window, pt1, pt2, (255, 255, 255), 2)  # White lines

    # Draw the individual joint dots
    for pt in points:
        cv2.circle(blank_window, pt, 5, (0, 255, 0), -1)  # Green dots

    # Show the reconstructed hand framework
    cv2.putText(blank_window, f"Saved Frame Row: {index}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    cv2.imshow("CSV Landmark Visualizer", blank_window)
    
    # Wait for key press. If ESC (key 27) is pressed, exit loop
    key = cv2.waitKey(0) & 0xFF
    if key == 27: 
        break

cv2.destroyAllWindows()
print("👋 Visualizer closed.")
