import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import pandas as pd
from datetime import datetime
from mediapipe.framework.formats import landmark_pb2

angles_df = pd.read_csv("2.csv")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoseClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
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

model = PoseClassifier(num_classes=5)
model.load_state_dict(torch.load("best.pth", map_location=device))
model.to(device)
model.eval()

class_names = ["Downdog", "Plank", "Warrior2", "Modified_Tree", "Standard_Tree"]

def get_csv_class_name(model_class_name):
    mapping = {
        "Downdog": "downdog",
        "Plank": "plank",
        "Warrior2": "warrior2",
        "Modified_Tree": "modified_tree_pose",
        "Standard_Tree": "tree",
    }
    return mapping.get(model_class_name, model_class_name.lower())

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1, 1)
    return np.degrees(np.arccos(cosine_angle))

def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        return None
    return [[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]

def detect_class(lm):
    if lm is None:
        return "None", 0.0
    
    lm_flat = [coord for point in lm for coord in point]
    input_tensor = torch.tensor(lm_flat, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted].item()
        
        # Only accept if confidence > 75%
        if confidence < 0.75:
            return "None", confidence
        
        return class_names[predicted], confidence

def draw_pose(frame, lm, target_pose, detected_pose):
    if lm is None:
        return frame
    
    color = (0, 255, 0) if detected_pose == target_pose else (0, 0, 255)
    landmark_spec = mp_drawing.DrawingSpec(color=color, thickness=4, circle_radius=6)
    connection_spec = mp_drawing.DrawingSpec(color=color, thickness=3)
    
    mp_lm = landmark_pb2.LandmarkList()
    for x, y, z in lm:
        mp_lm.landmark.add(x=float(x), y=float(y), z=float(z))
    
    mp_drawing.draw_landmarks(
        frame, mp_lm, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_spec,
        connection_drawing_spec=connection_spec
    )
    return frame

def draw_angles(frame, lm, target_pose):
    if lm is None:
        return frame
    
    h, w, _ = frame.shape
    csv_class = get_csv_class_name(target_pose)
    pose_rows = angles_df[angles_df['class'] == csv_class]
    
    for _, row in pose_rows.iterrows():
        idx1 = int(row['body_part1'])
        idx2 = int(row['body_part_vertex'])
        idx3 = int(row['body_part3'])
        
        if idx1 >= len(lm) or idx2 >= len(lm) or idx3 >= len(lm):
            continue
        
        p1, p2, p3 = lm[idx1], lm[idx2], lm[idx3]
        angle = calculate_angle(p1, p2, p3)
        
        p2_xy = (int(p2[0] * w), int(p2[1] * h))
    
        angle_text = f"{int(angle)}"
        text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    
        cv2.rectangle(
            frame,
            (p2_xy[0] - 5, p2_xy[1] - text_size[1] - 5),
            (p2_xy[0] + text_size[0] + 5, p2_xy[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Angle text in cyan
        cv2.putText(
            frame, angle_text, p2_xy,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
    
    return frame

def draw_info_panel(frame, target_pose, detected_pose, round_num, rounds_total, timer, confidence):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"Target: {target_pose}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Detected: {detected_pose} ({confidence*100:.0f}%)", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    timer_text = f"Round: {round_num}/{rounds_total} | {int(timer)}s"
    cv2.putText(frame, timer_text, (w - 400, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    
    return frame

def show_message_overlay(frame, message, duration_sec=2, size='normal'):
    h, w, _ = frame.shape
    if size == 'small':
        box_h, font_scale, thickness = 80, 0.7, 2
    else:
        box_h, font_scale, thickness = 120, 1.0, 3

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (50, h // 2 - box_h // 2),
        (w - 50, h // 2 + box_h // 2),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h // 2 + text_size[1] // 2
    
    cv2.putText(
        frame, message,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        thickness
    )
    
    cv2.imshow("FlexiFit AI - Pose Trainer", frame)
    cv2.waitKey(duration_sec * 1000)

def show_countdown_overlay(frame, count):
    h, w, _ = frame.shape
    
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (w // 2 - 100, h // 2 - 100),
        (w // 2 + 100, h // 2 + 100),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv2.putText(
        frame, str(count),
        (w // 2 - 40, h // 2 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        4, (0, 255, 255), 6
    )
    
    cv2.imshow("FlexiFit AI - Pose Trainer", frame)
    cv2.waitKey(1000)

def main(target_pose):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found!")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_video_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    rounds_total = 2
    duration = 5  
    round_num = 1
    round_start = None
    fail_count = 0
    fail_threshold = 10  
    switched_to_modified = False
    started_with_standard = (target_pose == "Standard_Tree")
    total_start = time.time()
    
    # Show initial countdown
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if ret:
            show_message_overlay(frame, f"Starting in {i}...", duration_sec=1, size='normal')
            out.write(frame)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        lm = extract_landmarks(frame)
        detected_pose, confidence = detect_class(lm)
        
        # Calculate timer
        timer = max(0, duration - (time.time() - round_start)) if round_start else duration
        
        frame = draw_info_panel(frame, target_pose, detected_pose, round_num, rounds_total, timer, confidence)
        frame = draw_pose(frame, lm, target_pose, detected_pose)
        frame = draw_angles(frame, lm, target_pose)
        
        if lm is None:
            cv2.putText(frame, "No person detected!", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            fail_count += 1
        
        elif detected_pose == target_pose:
            fail_count = 0  
            
            if round_start is None:
                round_start = time.time()
            
            elapsed = time.time() - round_start
            remaining = int(duration - elapsed)
            
            cv2.putText(frame, f"PERFECT! HOLD FOR {remaining}s", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Check if duration completed
            if elapsed >= duration:
                round_num += 1
                round_start = None
                
                if round_num <= rounds_total:
                    # Round complete, show countdown
                    for c in [3, 2, 1]:
                        show_countdown_overlay(frame, c)
                        out.write(frame)
                    
                    show_message_overlay(frame, "Get ready for next round!", duration_sec=2)
                    out.write(frame)
                
                else:
                    # All rounds complete OR need to switch
                    if started_with_standard and not switched_to_modified:
                        # Switch to modified tree
                        show_message_overlay(frame, "Great! Now trying Modified Tree...", 
                                           duration_sec=2, size='normal')
                        out.write(frame)
                        
                        target_pose = "Modified_Tree"
                        round_num = 1
                        switched_to_modified = True
                        round_start = None
                        fail_count = 0
                        continue
                    
                    else:
                      
                        total_time = int(time.time() - total_start)
                        final_message = f"Session Complete! Time: {total_time}s"
                        
                        show_message_overlay(frame, final_message, duration_sec=3, size='normal')
                        out.write(frame)
                        
                        break
        
        else:
            # Incorrect pose
            round_start = None
            fail_count += 1
            
            cv2.putText(frame, "Adjust your pose!", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Check if need to switch to modified (only for standard tree)
            if started_with_standard and not switched_to_modified and fail_count >= fail_threshold:
                show_message_overlay(frame, "Switching to Modified Tree in 3...", duration_sec=1, size='small')
                out.write(frame)
                show_message_overlay(frame, "2...", duration_sec=1, size='small')
                out.write(frame)
                show_message_overlay(frame, "1...", duration_sec=1, size='small')
                out.write(frame)
                
                target_pose = "Modified_Tree"
                round_num = 1
                fail_count = 0
                switched_to_modified = True
                round_start = None
        
        out.write(frame)
        
       
        cv2.imshow("FlexiFit AI - Pose Trainer", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    cap.release()
    out.release()
    cv2.destroyAllWindows()




target_pose = "Standard_Tree"  
    
main(target_pose)