# Helios
Uses OpenCV and MediaPipe to record hand landmarks, convert them into coordinate features, recognize gestures using Sklearn, and control your PC with those gestures via PyAutoGui.

## Project Status
This project has a limited dataset and therefore is not reliable for all environments. We recommend you gather your own dataset and use `data.py` to convert images to coordinates, then train the model again using `train.py`.

## Python Compatibility
This project is compatible with the following Python versions:
- **Supported:** 3.11 to 3.12  
- **Warning:** You may get errors on legacy versions (3.8 and below) as well as 3.13 due to C++ binary dependency conflicts.

## Installation
Install the required dependencies within your virtual environment:
```bash
pip install mediapipe opencv-python pandas scikit-learn pyautogui joblib numpy
