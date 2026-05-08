import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import joblib
import numpy as np
from collections import deque, Counter

# 1. LOAD MODEL
model = joblib.load('gestureModel.pkl')

# 2. SETUP DETECTOR (No Shortcuts)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 3. STABILITY PARAMETERS
prediction_buffer = deque(maxlen=6) # Store last 6 frames
CONFIDENCE_THRESHOLD = 0.85          # Model must be 85% sure to trust it
STABLE_GESTURE = "..."               # The currently confirmed gesture

def get_features(hand_landmarks):
    # Normalize landmarks relative to wrist (index 0)
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    
    max_value = max([abs(x) for x in features])
    if max_value > 0:
        features = [x / max_value for x in features]
    return features

# 4. START CAMERA
cap = cv2.VideoCapture(0)

print("System Active. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        features = get_features(detection_result.hand_landmarks[0])
        
        # Get probabilities
        probs = model.predict_proba([features])[0]
        max_idx = np.argmax(probs)
        confidence = probs[max_idx]
        predicted_label = model.classes_[max_idx]
        
        # Only add to buffer if we are confident
        if confidence > CONFIDENCE_THRESHOLD:
            prediction_buffer.append(predicted_label)
        
        # To "confirm" a gesture, the majority of the last 6 frames must match
        if len(prediction_buffer) == 6:
            counts = Counter(prediction_buffer)
            most_common, count = counts.most_common(1)[0]
            if count >= 5: # If 5 out of 6 frames match
                STABLE_GESTURE = most_common

        cv2.putText(frame, f"Gesture: {STABLE_GESTURE} ({int(confidence*100)}%)", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Hand lost? Clear buffer to reset
        prediction_buffer.clear()
        cv2.putText(frame, "No Hand Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Helios', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()  