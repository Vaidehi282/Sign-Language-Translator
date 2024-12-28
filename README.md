# Sign-Language-Translator
This project implements a real-time hand sign language recognition system using OpenCV, MediaPipe, and a trained Random Forest model. It detects hand landmarks, predicts corresponding letters (A-Z), and overlays the prediction on live camera frames.

## Features

- Detects and extracts hand landmarks in real-time using MediaPipe Hands.
- Predicts corresponding letters (A-Z) based on a Random Forest classifier trained on a custom dataset.
- Displays predictions on the live camera feed with bounding boxes and annotations.
- Provides a responsive and efficient real-time performance.

## Workflow

1. **Dataset Creation**: Created a custom dataset for all 26 English letters, capturing hand landmarks for training.
2. **Landmark Detection**: Used MediaPipe Hands to extract (x, y) coordinates of hand landmarks for feature generation.
3. **Model Training**: Trained a Random Forest model using extracted landmark features and saved it using pickle.
4. **Real-Time Prediction**: Integrated the trained model with OpenCV to process live video feed and predict letters dynamically.
