import sys
sys.path.append(".")

import cv2
import csv
import os
import time
import mediapipe as mp
from src.ingestion.webcam_stream import get_webcam_stream
from src.ingestion.pose_extractor import extract_pose_landmarks

mp_pose = mp.solutions.pose

CLASSES = ["safe", "medium_risk", "high_risk"]
SAVE_DIR = "data/raw"
FRAMES_PER_CLASS = 300  # ~10 seconds at 30fps

os.makedirs(SAVE_DIR, exist_ok=True)

def record_class(label, pose_model):
    output_path = os.path.join(SAVE_DIR, f"{label}.csv")
    print(f"\n>>> Get ready to pose: {label.upper()}")
    print("Starting in 3 seconds...")
    time.sleep(3)
    print("Recording... press 'q' to stop early.")

    rows = []
    count = 0

    for frame in get_webcam_stream():
        landmarks = extract_pose_landmarks(frame, pose_model)

        if landmarks is not None:
            row = landmarks.flatten().tolist()
            row.append(label)
            rows.append(row)
            count += 1
            print(f"  Frame {count}/{FRAMES_PER_CLASS}", end="\r")

        cv2.imshow(f"Recording: {label}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= FRAMES_PER_CLASS:
            break

    cv2.destroyAllWindows()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nSaved {count} frames to {output_path}")

with mp_pose.Pose() as pose_model:
    for label in CLASSES:
        record_class(label, pose_model)

print("\nAll classes recorded.")
