import cv2
import csv
import mediapipe as mp
from pathlib import Path

# ============================================================
# 1. PATH (Fixed "model" folder name matching your workspace)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "hand_landmarker.task"  # Changed from "models" to "model"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. MEDIAPIPE SETUP
# ============================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# ============================================================
# 3. SETTINGS FOR STATIC SYMBOLS
# ============================================================
LABEL = "J"
CSV_PATH = DATA_DIR / f"{LABEL}.csv"

# Changed running mode to IMAGE for reliable static symbol capturing
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# 4. CSV DATASET SETUP
# ============================================================
file_exists = CSV_PATH.exists()
csv_file = open(CSV_PATH, "a", newline="")
writer = csv.writer(csv_file)

if not file_exists:
    header = ["label"]
    for i in range(21):
        header.append(f"x{i}")
        header.append(f"y{i}")
        header.append(f"z{i}")
    writer.writerow(header)

# ============================================================
# 5. CAMERA INITIALIZATION
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera could not be opened")
    csv_file.close()
    exit()

print("===================================")
print("  Static Sign Dataset Collector    ")
print("===================================")
print(f"Current Target Symbol: {LABEL}")
print("Hold your hand steady in position!")
print("Press S → Save static landmark")
print("Press Q → Quit and close")
print("===================================")

# ============================================================
# 6. RUN DETECTOR LOOP
# ============================================================
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Frame not found")
            break

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect landmarks directly from the frame instance
        detection_result = landmarker.detect(mp_image)

        # Draw current live hand dots on the screen
        if detection_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # User Interface Text Overlay
        cv2.putText(frame, f"Label: {LABEL}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press S = Save Point | Q = Quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF

         # SAVE LANDMARK (Instantly grabs the exact static gesture)
        if key == ord("s"):
            if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                hand = detection_result.hand_landmarks[0]
                
                # 1. SAFELY EXTRACT HANDEDNESS FIRST
                # MediaPipe structures this as a list of lists: handedness[hand_idx][category_idx]
                hand_handedness = detection_result.handedness[0][0].category_name
                
                row = [LABEL]
                for landmark in hand:
                    # 2. CREATE A TEMPORARY COORDINATE OBJECT TO FLIP SAFELY
                    # Modifying landmark.x directly can break MediaPipe's tracking loop memory
                    x_coord = landmark.x
                    if hand_handedness == "Left":
                        x_coord = 1.0 - x_coord
                        
                    row.append(x_coord)
                    row.append(landmark.y)
                    row.append(landmark.z)
                
                writer.writerow(row)
                csv_file.flush()
                print(f"✅ Static coordinate for symbol '{LABEL}' successfully saved! ({hand_handedness} hand)")
            else:
                print("⚠️ Hand not detected! Make sure your hand is visible in the frame.")


        # QUIT APPLICATION
        if key == ord("q"):
            break

# ============================================================
# 7. CLEANUP RESOURCE REGION
# ============================================================
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("👋 Dataset collection stopped safely.")
