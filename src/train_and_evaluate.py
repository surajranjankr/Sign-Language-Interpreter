import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. THE BRAIN: Target actions
actions = np.array(['before', 'cool', 'thin', 'drink', 'go', 'computer', 'cousin', 'help', 'who', 'accident', 'bed', 'bowling', 'candy', 'short', 'tall', 'thanksgiving', 'trade', 'basketball', 'call', 'change'])
label_map = {label:num for num, label in enumerate(actions)}

import json

def find_all_landmarks():
    sequences, labels = [], []
    
    # 1. FIND THE MAP (FiftyOne usually names this samples.json)
    map_file = None
    for root, dirs, files in os.walk("data"):
        if "samples.json" in files:
            map_file = os.path.join(root, "samples.json")
            break
    
    if not map_file:
        print("❌ Could not find samples.json. We don't know which file is which sign!")
        return np.array([]), np.array([])

    with open(map_file, 'r') as f:
        metadata = json.load(f)

    # 2. MATCH FILES USING THE METADATA
    # Metadata structure: samples -> [{filepath: "...", gloss: {label: "..."}}]
    for sample in metadata['samples']:
        label = sample['gloss']['label'].lower()
        if label in actions:
            # Convert the metadata filepath to your local .npy path
            file_id = os.path.basename(sample['filepath']).split('.')[0]
            
            # Find the actual .npy files (original, jitter, scale) for this ID
            for root, dirs, files in os.walk("data"):
                for f in files:
                    if f.startswith(file_id) and f.endswith(".npy"):
                        res = np.load(os.path.join(root, f))
                        sequences.append(res)
                        labels.append(label_map[label])

    print(f"✅ Successfully mapped {len(sequences)} sequences using samples.json!")
    return np.array(sequences), np.array(labels)

# --- EXECUTION ---
X_raw, y_raw = find_all_landmarks()

if len(X_raw) > 0:
    y = to_categorical(y_raw).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

    # Bi-LSTM Architecture
    model = Sequential([
        Input(shape=(30, 63)),
        Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=False, activation='relu')),
        Dense(64, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("\n🚀 Training Starting...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

    # EVALUATION
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat_classes = np.argmax(yhat, axis=1).tolist()

    print("\n📊 FINAL CLASSIFICATION REPORT:")
    print(classification_report(ytrue, yhat_classes, target_names=actions))
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(ytrue, yhat_classes), annot=True, xticklabels=actions, yticklabels=actions, cmap='Greens')
    plt.show()
else:
    print("❌ Error: Even with deep-search, no .npy files were found. Did 'extract_landmarks.py' definitely run?")

model.save('models/asl_model.h5')