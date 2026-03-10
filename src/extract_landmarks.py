import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Download this first: wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
model_path = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_videos", "data")
LANDMARK_DIR = os.path.join(BASE_DIR, "data", "landmarks")
os.makedirs(LANDMARK_DIR, exist_ok=True)

def get_sequence_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect landmarks
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            # Flatten 21 points (x, y, z) into a 63-element array
            lms = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.hand_landmarks[0]]).flatten()
            sequence.append(lms)
        else:
            # If hand is missing, use zeros (maintains sequence timing)
            sequence.append(np.zeros(21 * 3))
            
    cap.release()

    # --- THE 30-FRAME STANDARDIZER ---
    # We need a fixed input shape for the LSTM: (30, 63)
    if len(sequence) < 30:
        # Pad with zeros if video is too short
        padding = [np.zeros(63)] * (30 - len(sequence))
        sequence.extend(padding)
    
    return np.array(sequence[:30]) # Truncate if too long

# --- THE AUGMENTATION ENGINE ---
def save_variants(data, base_filename):
    # Original
    np.save(os.path.join(LANDMARK_DIR, f"{base_filename}_orig.npy"), data)
    
    # Add Jitter (Simulates shaky hands)
    noise = np.random.normal(0, 0.003, data.shape)
    np.save(os.path.join(LANDMARK_DIR, f"{base_filename}_jitter.npy"), data + noise)
    
    # Random Scaling (Simulates moving closer/further)
    scale = np.random.uniform(0.85, 1.15)
    np.save(os.path.join(LANDMARK_DIR, f"{base_filename}_scale.npy"), data * scale)

# --- MAIN EXECUTION ---
video_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".mp4")]
print(f"🚀 Tasks API Online. Processing {len(video_files)} videos...")

for i, vid_name in enumerate(video_files):
    vid_path = os.path.join(RAW_DATA_DIR, vid_name)
    base_name = os.path.splitext(vid_name)[0]
    
    try:
        landmarks = get_sequence_landmarks(vid_path)
        save_variants(landmarks, base_name)
        
        if (i + 1) % 20 == 0:
            print(f"✅ Processed {i+1}/{len(video_files)} videos...")
            
    except Exception as e:
        print(f"⚠️ Error processing {vid_name}: {e}")

print(f"🏁 DONE! Check {LANDMARK_DIR} for your .npy files.")
print(f"📊 Total files generated: {len(os.listdir(LANDMARK_DIR))}")