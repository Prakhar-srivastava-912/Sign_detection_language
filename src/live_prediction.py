
from pathlib import Path
import cv2
import joblib
import pandas as pd
import mediapipe as mp
import pyttsx3
import threading
import queue

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "sign_model.pkl"
TASK_PATH = PROJECT_DIR / "model" / "hand_landmarker.task"

model = joblib.load(MODEL_PATH)

# Speech system
speech_queue = queue.Queue()


def speech_worker():
    while True:
        message = speech_queue.get()

        try:
            if message is None:
                break

            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(str(message))
            engine.runAndWait()
            engine.stop()
            del engine

        except Exception as error:
            print("Speech error:", error)

        finally:
            speech_queue.task_done()


threading.Thread(target=speech_worker, daemon=True).start()

# MediaPipe setup
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(TASK_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

camera = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.30

current_prediction = ""
word = ""

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while camera.isOpened():
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int(
            cv2.getTickCount() * 1000 / cv2.getTickFrequency()
        )

        result = landmarker.detect_for_video(mp_image, timestamp)

        current_prediction = ""

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

           

            # Prepare 63 features in the model's expected order
            features = []

            for point in hand:
                features.extend([point.x, point.y, point.z])

            columns = [
                f"{axis}{i}"
                for i in range(21)
                for axis in ["x", "y", "z"]
            ]

            input_data = pd.DataFrame(
                [features], columns=columns
            )

            probabilities = model.predict_proba(input_data)[0]
            best_index = probabilities.argmax()
            confidence = probabilities[best_index]

            prediction = str(model.classes_[best_index])

            if confidence >= CONFIDENCE_THRESHOLD:
                current_prediction = prediction

                cv2.putText(
                    frame,
                    f"Sign: {current_prediction}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                ) 
        # Display the word being built
        cv2.putText(
            frame,
            f"Word: {word}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        # Instructions
        instructions = [
            "ENTER: Add detected letter",
            "SPACE: Add space",
            "S: Speak word",
            "BACKSPACE: Delete last character",
            "C: Clear word",
            "Q: Quit"
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (20, 200 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

        cv2.imshow("Sign Language Word Builder", frame)

        key = cv2.waitKey(1) & 0xFF

        # ENTER: add the currently detected letter
        if key in (10, 13):
            if current_prediction:
                word += current_prediction
                print("Current word:", word)

        # SPACE: separate words
        elif key == ord(" "):
            if word and not word.endswith(" "):
                word += " "

        # S: speak the whole word or sentence
        elif key == ord("s"):
            text_to_speak = word.strip()

            if text_to_speak:
                speech_queue.put(text_to_speak)

        # BACKSPACE: remove the last character
        elif key in (8, 127):
            word = word[:-1]

        # C: clear everything
        elif key == ord("c"):
            word = ""

        # Q: quit
        elif key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
