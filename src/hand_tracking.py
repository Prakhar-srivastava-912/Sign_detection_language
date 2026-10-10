import cv2
import mediapipe as mp
from pathlib import Path
import time

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = PROJECT_ROOT / "model" / "hand_landmarker.task"

# MediaPipe options
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

# Create Hand Landmarker
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

# Check camera
if not cap.isOpened():
    print("❌ Camera is not opened")
    exit()

print("✅ Camera started!")
print("Press 'q' to quit")

while True:
    success, frame = cap.read()

    if not success:
        print("❌ Camera frame is not read")
        break

    # Flip image horizontally
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert image to MediaPipe format
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp for VIDEO mode
    timestamp_ms = int(time.time() * 1000)

    # Detect hands
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # Get frame dimensions
    height, width, _ = frame.shape

    # Draw landmarks and process mirroring
    if result.hand_landmarks:
        # Loop through both the landmarks array and the handedness labels array at the same time
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            
            # Get whether this specific hand is "Left" or "Right"
            hand_handedness = result.handedness[idx][0].category_name

            for landmark in hand_landmarks:
                
                # Keep a copy of the original X coordinate for drawing on screen
                x_coord = landmark.x
                
                # ====================================================
                # MIRROR LOGIC FOR AI PREDICTION (Later Phase)
                # ====================================================
                # If you need to send this coordinate to a model trained 
                # only on right hands, use the mirrored value:
                if hand_handedness == "Left":
                    mirrored_x = 1.0 - x_coord
                else:
                    mirrored_x = x_coord
                # ====================================================

                # Draw the dots safely on the visible screen using the display coordinate
                x = int(x_coord * width)
                y = int(landmark.y * height)

                # Set color: Blue for Left hand, Green for Right hand so you can see it working!
                dot_color = (255, 0, 0) if hand_handedness == "Left" else (0, 255, 0)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    dot_color,
                    -1
                )

    # Show result
    cv2.imshow("Hand Tracking", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
detector.close()
