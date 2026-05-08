import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pyautogui
import joblib
import numpy as np
import time
from collections import deque, Counter

# --- SETUP ---
pyautogui.FAILSAFE = True
screen_width, screen_height = pyautogui.size()
smoothing = 0.3 
curr_x, curr_y = screen_width // 2, screen_height // 2
last_action_time = 0
COOLDOWN = 0.6 

model = joblib.load('gestureModel.pkl')

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)


prediction_buffer = deque(maxlen=6) 
CONFIDENCE_THRESHOLD = 0.85          
STABLE_GESTURE = "..."               

def get_features(landmarks_list):
    base_x = landmarks_list[0].x
    base_y = landmarks_list[0].y
    base_z = landmarks_list[0].z
    
    features = []
    for lm in landmarks_list:
        features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    
    max_value = max([abs(x) for x in features])
    if max_value > 0:
        features = [x / max_value for x in features]
    return features

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

print("System Active. Move palm to steer cursor. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        features = get_features(hand_landmarks)
        probs = model.predict_proba([features])[0]
        max_idx = np.argmax(probs)
        confidence = probs[max_idx]
        predicted_label = model.classes_[max_idx]
    
        if confidence > CONFIDENCE_THRESHOLD:
            prediction_buffer.append(predicted_label)
        
        if len(prediction_buffer) == 6:
            counts = Counter(prediction_buffer)
            most_common, count = counts.most_common(1)[0]
            if count >= 4: STABLE_GESTURE = most_common

        wrist = hand_landmarks[0]
        target_x = int(wrist.x * screen_width)
        target_y = int(wrist.y * screen_height)
        curr_x = int(curr_x + (target_x - curr_x) * smoothing)
        curr_y = int(curr_y + (target_y - curr_y) * smoothing)
        
        if STABLE_GESTURE == "open":
            pyautogui.moveTo(curr_x, curr_y)
        elif STABLE_GESTURE == "click":
            if time.time() - last_action_time > COOLDOWN:
                pyautogui.click()
                last_action_time = time.time()
        elif STABLE_GESTURE == "rightClick":
            if time.time() - last_action_time > COOLDOWN:
                pyautogui.rightClick()
                last_action_time = time.time()

        cv2.putText(frame, f"Gesture: {STABLE_GESTURE}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        prediction_buffer.clear()
        cv2.putText(frame, "No Hand Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Helios', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()