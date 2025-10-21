"""
Pose Detection and Classification on Video using MediaPipe + PyTorch
-------------------------------------------------------------------
This script:
1. Extracts human pose landmarks from each frame of a video using MediaPipe.
2. Classifies the pose using a trained PyTorch neural network.
3. Highlights frames based on whether the detected pose matches the target.
4. Saves the annotated video to a new file.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp


# MEDIAPIPE SETUP
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# POSE CLASSIFIER MODEL
class PoseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PoseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(99, 128),    
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),   
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)  
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = PoseClassifier(num_classes=5)
model.load_state_dict(torch.load("best.pth", map_location=device))
model.to(device)
model.eval()

class_names = ["Downdog", "Plank", "Warrior2", "Modified_Tree", "Standard_Tree"]


def extract_landmarks(frame, pose_detector):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(frame_rgb)
    if not results.pose_landmarks:
        return None, None

    landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
    return landmarks, results.pose_landmarks


def detect_class(landmarks):
   
    lm_flat = [coord for point in landmarks for coord in point]
    input_tensor = torch.tensor(lm_flat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return class_names[predicted_class]


def analyze_video(video_path, target_pose, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_pose.Pose(static_image_mode=False) as pose_detector:
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            landmarks, pose_landmarks = extract_landmarks(frame, pose_detector)

            if landmarks is not None:
                class_name = detect_class(landmarks)
                color = (0, 255, 0) if class_name == target_pose else (0, 0, 255)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )

                cv2.putText(frame, f"Pose: {class_name}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

            out.write(frame)


    cap.release()
    out.release()



VIDEO_PATH = "/content/plank.mp4"
TARGET_POSE = "Plank"
OUTPUT_PATH = "analyzed_output.mp4"
analyze_video(VIDEO_PATH, TARGET_POSE, OUTPUT_PATH)
