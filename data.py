import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os

# 1. Initialize the Detector (The official Google Tasks API way)
# This bypasses mp.solutions entirely
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def keypoints(path):
    # Load image using MediaPipe's own tool
    mp_image = mp.Image.create_from_file(path)
    # Detect hands
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        # Get the first hand detected
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # 'hand_landmarks' here is a list of NormalizedLandmark objects (x, y, z)
        # We perform the same normalization (relative to wrist)
        base_x = hand_landmarks[0].x
        base_y = hand_landmarks[0].y
        base_z = hand_landmarks[0].z
        
        features = []
        for lm in hand_landmarks:
            features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            
        # Scale Invariance
        max_value = max([abs(x) for x in features])
        if max_value > 0:
            features = [x / max_value for x in features]
        return features
    return None

# 2. Define Columns
columns = ['label']
for i in range(21):
    columns.extend([f'x{i}', f'y{i}', f'z{i}'])

# 3. Initialize DataFrame
df = pd.DataFrame(columns=columns)
    
def main():
    folders = {'Click': 'click', 'Right Click': 'rightClick', 'Open': 'open'}
    
    for folder, label in folders.items():
        if not os.path.exists(folder):
            print(f"Skipping {folder} (folder not found)")
            continue
            
        print(f"Processing {folder}...")
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                features = keypoints(os.path.join(folder, file))
                if features:
                    df.loc[len(df)] = [label] + features
    
    # 4. Save to CSV
    df.to_csv('trainData.csv', index=False)
    print(f"Done! Saved to trainData.csv. Total rows: {len(df)}")

if __name__ == "__main__":
    main()