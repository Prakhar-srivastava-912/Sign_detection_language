from pathlib import Path
import mediapipe as mp

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Model file path
MODEL_PATH = PROJECT_ROOT / "model" / "hand_landmarker.task"

print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Hand Landmarker setup
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2
)

# Load Hand Landmarker
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

print("✅ Hand Landmarker loaded successfully!")