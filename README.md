# Sign Language Detection

## About the Project
This project uses Python, OpenCV, and MediaPipe to detect hand landmarks and recognize hand signs through a webcam. It includes data collection, model training, visualization, and live prediction.

## Features
- Hand landmark detection using MediaPipe
- Custom dataset collection and CSV storage
- Machine learning model training
- Real-time sign prediction using a webcam
- Text-to-speech output for detected words
- Word building using keyboard controls

## Technologies Used
- Python
- OpenCV
- MediaPipe
- Pandas
- Scikit-learn
- Joblib
- pyttsx3

## Project Structure
Mini project/
├── data/
│   └── raw/
├── model/
│   ├── hand_landmarker.task
│   └── sign_model.pkl
├── src/
│   ├── collect_data.py
│   ├── hand_tracking.py
│   ├── live_prediction.py
│   ├── train_model.py
│   └── visualize_csv.py
├── camera_test.py
├── test_model.py
├── .gitignore
└── README.md

## Setup
1. Install Python.
2. Clone this repository:
   `git clone https://github.com/Prakhar-srivastava-912/Sign_detection_language.git`
3. Open the project folder.
4. Install the required Python libraries using `requirements.txt` once it is available.
5. Run the appropriate Python script from the project folder.

## Usage
- Use `collect_data.py` to collect hand landmark samples.
- Use `train_model.py` to train the recognition model.
- Use `live_prediction.py` for real-time prediction.
- Follow the keyboard instructions displayed by the live prediction program.

## Notes
- A webcam is required for live prediction.
- Recognition accuracy depends on the training data, lighting, and hand positioning.
- Dynamic signs such as J and Z may require movement-based recognition.

## Author
Prakhar Srivastava
