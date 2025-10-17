import streamlit as st
import cv2
import numpy as np
import torch
import base64
import torch.nn as nn
import mediapipe as mp
from elasticsearch import Elasticsearch
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
import tempfile
import os
import warnings
from datetime import datetime
from PIL import Image
import json
from collections import Counter

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="FlexiFit AI - PCOS/PCOD Exercise Coach",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
   <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap" rel="stylesheet">
   """, unsafe_allow_html=True)

# ==========================================
# CUSTOM CSS - BEAUTIFUL LIGHT THEME WITH #DEE276
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Beautiful gradient background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 50%, #fefff5 100%);
    }
    
    .main {
        padding: 2rem 1.5rem;
    }
    
    /* Main header with brand color */

.main-header { 
    font-size: 3rem;
    font-weight: 900; 
    color: #919c08;
    text-shadow: 0 4px 20px rgba(222, 226, 118, 0.3); 
    margin: 0;
    padding: 0;
    line-height: 1.2;
}


    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        background: #919c08;;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        font-weight: 500;
        animation: fadeInUp 1s ease-out;
        letter-spacing: 2px;
    }

    /* Glassmorphism cards with brand color accents */
    .glass-card {
        background: linear-gradient(135deg, #ffffff 0%, #fefff5 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 2px solid rgba(222, 226, 118, 0.3);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(222, 226, 118, 0.2),
                    0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(222, 226, 118, 0.08) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }

    .glass-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 40px rgba(222, 226, 118, 0.35),
                    0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: rgba(222, 226, 118, 0.5);
    }

    /* Exercise cards */
    .exercise-card {
        background: linear-gradient(135deg, #ffffff 0%, #fefff8 100%);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 18px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(222, 226, 118, 0.15),
                    0 2px 8px rgba(0, 0, 0, 0.05);
        border: 2px solid rgba(222, 226, 118, 0.25);
        transition: all 0.3s ease;
        position: relative;
    }

    .exercise-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(222, 226, 118, 0.3),
                    0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: rgba(222, 226, 118, 0.4);
    }

    /* Success box with brand color */
    .success-box {
        background: linear-gradient(135deg, rgba(222, 226, 118, 0.12) 0%, rgba(240, 238, 154, 0.08) 100%);
        backdrop-filter: blur(15px);
        border: none;
        border-left: 5px solid #dee276;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(222, 226, 118, 0.2),
                    0 2px 8px rgba(0, 0, 0, 0.05);
        animation: slideInLeft 0.6s ease-out;
        color: #2c3e50;
    }

    .info-box {
        background: linear-gradient(135deg, rgba(222, 226, 118, 0.08) 0%, rgba(240, 238, 154, 0.05) 100%);
        backdrop-filter: blur(15px);
        border: none;
        border-left: 5px solid #dee276;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(222, 226, 118, 0.15),
                    0 2px 8px rgba(0, 0, 0, 0.05);
        color: #2c3e50;
    }

    /* Beautiful buttons with brand color */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.2rem;
        font-weight: 700;
        font-size: 1rem;
        border: 2px solid #dee276;
        background: linear-gradient(135deg, #dee276 0%, #eef18d 100%);
        backdrop-filter: blur(10px);
        color: #2c3e50;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(222, 226, 118, 0.3),
                    0 2px 6px rgba(0, 0, 0, 0.08);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        position: relative;
        overflow: hidden;
    }

    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.5);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }

    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(222, 226, 118, 0.5),
                    0 4px 12px rgba(0, 0, 0, 0.12);
        background: linear-gradient(135deg, #c5cc3d 0%, #dee276 100%);
        color: #1a1a2e;
        border-color: #c5cc3d;
    }

    /* Metric cards with brand color */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #fffef5 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: #2c3e50;
        padding: 2rem;
        border-radius: 18px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(222, 226, 118, 0.25),
                    0 2px 8px rgba(0, 0, 0, 0.05);
        border: 2px solid rgba(222, 226, 118, 0.3);
        transition: all 0.4s ease;
        animation: fadeIn 1s ease-out;
        position: relative;
        overflow: hidden;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(222, 226, 118, 0.15) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 15px 50px rgba(222, 226, 118, 0.4),
                    0 6px 16px rgba(0, 0, 0, 0.1);
        border-color: #dee276;
    }

    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #dee276;
    }

    .metric-card h1 {
        margin: 1rem 0 0 0;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #dee276, #c5cc3d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

 .video-container {
  width: 400px;
  height: 400px;
  margin: 1.5rem auto;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 15px 50px rgba(222, 226, 118, 0.3),
              0 4px 12px rgba(0, 0, 0, 0.08);
  border: 3px solid rgba(222, 226, 118, 0.5);
  background: #ffffff;
  backdrop-filter: blur(10px);
  display: flex;
  justify-content: center;
  align-items: center;
}

.video-container video {
  width: 100%;
  height: 100%;
  object-fit: cover; /* crop and fill container */
}


    /* Chat messages with updated colors */
    .chat-user {
        background: linear-gradient(135deg, rgba(222, 226, 118, 0.15) 0%, rgba(240, 238, 154, 0.1) 100%);
        backdrop-filter: blur(15px);
        padding: 1.5rem;
        border-radius: 18px 18px 5px 18px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(222, 226, 118, 0.2),
                    0 2px 6px rgba(0, 0, 0, 0.05);
        animation: slideInRight 0.4s ease-out;
        font-size: 1rem;
        border: 2px solid rgba(222, 226, 118, 0.3);
        color: #2c3e50;
    }

    .chat-assistant {
        background: linear-gradient(135deg, #dee276 0%, #eef18d 100%);
        backdrop-filter: blur(15px);
        padding: 1.5rem;
        border-radius: 18px 18px 18px 5px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(222, 226, 118, 0.25),
                    0 2px 6px rgba(0, 0, 0, 0.05);
        animation: slideInLeft 0.4s ease-out;
        font-size: 1rem;
        border: 2px solid rgba(197, 204, 61, 0.3);
        color: #2c3e50;
    }

    /* Tabs with brand color */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #fffef8 100%);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid rgba(222, 226, 118, 0.3);
        box-shadow: 0 4px 15px rgba(222, 226, 118, 0.15),
                    0 2px 6px rgba(0, 0, 0, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0 2rem;
        font-size: 0.95rem;
        font-weight: 700;
        border-radius: 12px;
        color: #666;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #dee276 0%, #eef18d 100%);
        backdrop-filter: blur(15px);
        color: #2c3e50;
        box-shadow: 0 4px 20px rgba(222, 226, 118, 0.4);
        border: 2px solid rgba(222, 226, 118, 0.4);
    }

    /* Progress bar with brand color */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #dee276 0%, #eef18d 100%);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(222, 226, 118, 0.5);
    }

    /* Input fields with brand color accents */
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid rgba(222, 226, 118, 0.3);
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: #ffffff;
        backdrop-filter: blur(10px);
        color: #2c3e50;
    }

    .stTextInput>div>div>input:focus {
        border-color: #dee276;
        box-shadow: 0 0 0 4px rgba(222, 226, 118, 0.2);
        background: #ffffff;
    }

    .streamlit-expanderHeader {
        background: rgba(222, 226, 118, 0.12);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        font-weight: 700;
        color: #2c3e50;
        font-size: 1rem;
        border: 1px solid rgba(222, 226, 118, 0.3);
    }

    /* Sidebar with brand color gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #fefff5 100%);
        color: #1C1C1C;
        border-right: 3px solid rgba(222, 226, 118, 0.4);
    }

    [data-testid="stSidebar"] .element-container {
        color: #1C1C1C;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1C1C1C;
    }

    /* Badges with brand color */
    .badge-primary {
        background: linear-gradient(135deg, rgba(222, 226, 118, 0.3) 0%, rgba(240, 238, 154, 0.2) 100%);
        color: #2c3e50;
        border: 1px solid rgba(222, 226, 118, 0.4);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.3rem;
    }

    .badge-success {
        background: linear-gradient(135deg, rgba(222, 226, 118, 0.35) 0%, rgba(240, 238, 154, 0.25) 100%);
        color: #2c3e50;
        border: 1px solid rgba(222, 226, 118, 0.5);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.3rem;
    }

    .badge-warning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 235, 59, 0.15) 100%);
        color: #2c3e50;
        border: 1px solid rgba(255, 193, 7, 0.4);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.3rem;
    }

    /* Typography */
    h1, h2, h3 {
        color: #2c3e50;
    }

    p, li, span {
        font-size: 1rem;
        color: #2c3e50;
        line-height: 1.7;
    }

    label {
        color: #2c3e50 !important;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Icon colors */
    .icon-primary {
        color: #919c08;
        font-size: 1.2rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }

    .icon-accent {
        color: #919c08;
        font-size: 1.2rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }

    /* Result display with brand colors */
    .result-header {
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #dee276 0%, #c5cc3d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 2rem 0;
        animation: fadeIn 0.8s ease-out;
    }

    .result-status {
        padding: 2rem;
        border-radius: 18px;
        margin: 2rem 0;
        animation: slideInUp 0.6s ease-out;
    }

    .result-perfect {
        background: linear-gradient(135deg, rgba(222, 226, 118, 0.25) 0%, rgba(240, 238, 154, 0.18) 100%);
        border: 2px solid rgba(222, 226, 118, 0.5);
        color: #2c3e50;
        box-shadow: 0 10px 40px rgba(222, 226, 118, 0.4);
    }
    
    .result-mismatch {
        background: linear-gradient(135deg, rgba(255, 235, 238, 0.8) 0%, rgba(255, 245, 245, 0.6) 100%);
        border: 2px solid rgba(255, 107, 107, 0.4);
        color: #2c3e50;
        box-shadow: 0 10px 40px rgba(255, 107, 107, 0.3);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'exercise_history' not in st.session_state:
    st.session_state.exercise_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analyzed_video_path' not in st.session_state:
    st.session_state.analyzed_video_path = None

# ==========================================
# MODEL CLASSES AND LOADING
# ==========================================
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

@st.cache_resource
def load_pose_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseClassifier(num_classes=5)
    model.load_state_dict(torch.load("best.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def setup_elasticsearch():
    es = Elasticsearch(
        cloud_id=os.environ["ES_CLOUD_ID"],
        api_key=os.environ["ES_API_KEY"]
    )
    return es

@st.cache_resource
def setup_vertex_ai():
    b64_string = os.environ["VERTEX_SERVICE_ACCOUNT_B64"]
    service_account_json = base64.b64decode(b64_string)
    service_account_info = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    aiplatform.init(
        project=os.environ["VERTEX_PROJECT_ID"],
        location=os.environ["VERTEX_LOCATION"],
        credentials=credentials
    )
    return GenerativeModel("gemini-2.5-flash")

# Load resources
model, device = load_pose_model()
es = setup_elasticsearch()
gemini_model = setup_vertex_ai()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class_names = ["Downdog", "Plank", "Warrior2", "Modified_Tree", "Standard_Tree"]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_landmarks(frame):
    """Extract pose landmarks from frame"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=False) as pose_detector:
        results = pose_detector.process(frame_rgb)
        if not results.pose_landmarks:
            return None, None
        lm = results.pose_landmarks.landmark
        return [[l.x, l.y, l.z] for l in lm], results

def detect_class(lm):
    """Detect exercise class from landmarks"""
    lm_flat = [coord for point in lm for coord in point]
    input_tensor = torch.tensor(lm_flat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted].item()
        return class_names[predicted], confidence

def analyze_video(video_path, target_pose):
    """Analyze video and detect poses"""
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    detected_poses = []
    confidences = []
    frame_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing frame {frame_count}/{total_frames}...")
        
        lm, results = extract_landmarks(frame)
        
        if lm is not None:
            class_name, confidence = detect_class(lm)
            detected_poses.append(class_name)
            confidences.append(confidence)
            
            is_correct = (class_name == target_pose)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=4, circle_radius=4),
                mp_drawing.DrawingSpec(color=color, thickness=4)
            )
            
            cv2.putText(frame, f"Target: {target_pose}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Detected: {class_name}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            status = "CORRECT ✓" if is_correct else "INCORRECT ✗"
            status_color = (0, 255, 0) if is_correct else (0, 0, 255)
            cv2.putText(frame, status, (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 4)
        
        out.write(frame)
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    if detected_poses:
        most_common = Counter(detected_poses).most_common(1)[0]
        detected_pose = most_common[0]
        avg_confidence = np.mean(confidences)
        accuracy = (detected_poses.count(target_pose) / len(detected_poses)) * 100
        
        return {
            'output_path': output_path,
            'detected_pose': detected_pose,
            'confidence': avg_confidence,
            'accuracy': accuracy,
            'total_frames': frame_count,
            'match': detected_pose == target_pose
        }
    
    return None

def search_exercises(query):
    """Search exercises in Elasticsearch"""
    result = es.search(
        index="pcos_exercises",
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["name", "description", "pcos_benefits", "keywords"]
                }
            },
            "size": 10
        }
    )
    return [hit['_source'] for hit in result['hits']['hits']]

def get_all_exercises():
    """Get all exercises from Elasticsearch"""
    result = es.search(
        index="pcos_exercises",
        body={"query": {"match_all": {}}, "size": 10}
    )
    return [hit['_source'] for hit in result['hits']['hits']]

def chat_with_ai(user_message):
    """Chat with AI coach using Vertex AI"""
    exercises = get_all_exercises()
    exercise_context = "\n".join([f"- {ex['name']}: {ex['description']}" for ex in exercises])
    
    prompt = f"""You are a PCOS/PCOD exercise coach. Be helpful, encouraging, and specific.

Available exercises:
{exercise_context}

User's recent exercise history:
{json.dumps(st.session_state.exercise_history[-3:], indent=2)}

User asks: {user_message}

Provide a helpful, personalized response. If suggesting exercises, explain why they help with PCOS/PCOD."""
    
    response = gemini_model.generate_content(prompt)
    return response.text

def get_exercise_image_path(exercise_name, exercise_id):
    """Get the correct image path for an exercise"""
    exercise_name_lower = exercise_name.lower()
    
    if 'downward' in exercise_name_lower or 'down dog' in exercise_name_lower or 'downdog' in exercise_name_lower:
        return "Downdog.jpg"
    elif 'plank' in exercise_name_lower:
        return "Plank.jpg"
    elif 'warrior' in exercise_name_lower:
        return "Warrior2.jpg"
    elif 'modified' in exercise_name_lower and 'tree' in exercise_name_lower:
        return "Modified_Tree.jpg"
    elif 'tree' in exercise_name_lower:
        return "Standard_Tree.jpg"
    else:
        return f"{exercise_id}.jpg"

# ==========================================
# MAIN APP LAYOUT
# ==========================================
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; gap: 1.5rem; margin: 2rem 0;">
    <img src="data:image/png;base64,{}" style="width: 100px; height: auto;">
    <h1 class="main-header">FLEXIFIT AI</h1>
</div>
""".format(base64.b64encode(open("logo.png", "rb").read()).decode()), unsafe_allow_html=True)


st.markdown('<p class="sub-header">Your AI-Powered PCOS/PCOD Exercise Coach with Real-Time Pose Detection</p>', unsafe_allow_html=True)

# ==========================================
# STATS ROW
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3><i class="fa-solid fa-dumbbell icon-primary"></i> Exercises</h3>
        <h1>{len(st.session_state.exercise_history)}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_accuracy = np.mean([h['accuracy'] for h in st.session_state.exercise_history]) if st.session_state.exercise_history else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3><i class="fa-solid fa-bullseye icon-primary"></i> Accuracy</h3>
        <h1>{avg_accuracy:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3><i class="fa-solid fa-comments icon-primary"></i> Messages</h3>
        <h1>{len(st.session_state.chat_history)}</h1>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3><i class="fa-solid fa-brain icon-primary"></i> AI Power</h3>
        <h1>92%</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Workout Tracker",
    "Analyze Video",
    "AI Coach Chat",
    "Progress Stats"
])

# ==========================================
# TAB 1: EXERCISE LIBRARY
# ==========================================
with tab1:
    st.markdown('<h2 class="result-header"><i class="fa-solid fa-book icon-primary"></i> PCOS/PCOD Exercise Library</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #262626; margin-bottom: 2rem;">Browse our curated collection of exercises specifically designed for PCOS/PCOD management</p>', unsafe_allow_html=True)
    
    search_query = st.text_input("Search exercises...", placeholder="Try: balance, beginner, stress relief, hormonal balance...")
    
    if search_query:
        exercises = search_exercises(search_query)
        
        if exercises:
            st.markdown(f'<h3 style="text-align: center; margin: 2rem 0;"><i class="fa-solid fa-bullseye icon-primary"></i> Found {len(exercises)} exercises</h3>', unsafe_allow_html=True)
            
            cols = st.columns(2)
            
            for idx, ex in enumerate(exercises):
                with cols[idx % 2]:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    
                    image_name = get_exercise_image_path(ex.get('name', ''), ex.get('exercise_id', ''))
                    
                    if os.path.exists(image_name):
                        img = Image.open(image_name)
                        img = img.resize((350, 350))
                        st.image(img, width=350)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, rgba(199, 208, 108, 0.15) 0%, rgba(224, 231, 134, 0.08) 100%); 
                                    height: 200px; display: flex; align-items: center; justify-content: center;
                                    border-radius: 12px; color: #919c08; font-size: 1.2rem; border: 2px solid rgba(224, 231, 134, 0.2);
                                    backdrop-filter: blur(10px);">
                            <i class="fa-solid fa-image icon-primary" style="font-size: 3rem;"></i>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"### <i class='fa-solid fa-heart-pulse icon-primary'></i> {ex['name']}", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <span class="badge badge-primary"><i class="fa-solid fa-layer-group icon-primary"></i> {ex['category']}</span>
                    <span class="badge badge-warning"><i class="fa-solid fa-signal icon-accent"></i> {ex['difficulty']}</span>
                    <span class="badge badge-success"><i class="fa-solid fa-clock icon-primary"></i> {ex['duration_seconds']}s</span>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**<i class='fa-solid fa-repeat icon-primary'></i> Reps:** {ex['reps']}", unsafe_allow_html=True)
                    
                    with st.expander(" View Details", expanded=False):
                        st.markdown(f"**<i class='fa-solid fa-info-circle icon-primary'></i> Description:**\n{ex['description']}", unsafe_allow_html=True)
                        st.markdown("**<i class='fa-solid fa-star icon-primary'></i> PCOS/PCOD Benefits:**", unsafe_allow_html=True)
                        for benefit in ex['pcos_benefits']:
                            st.markdown(f"• {benefit}", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 2rem;">
                <h3><i class="fa-solid fa-magnifying-glass icon-primary" style="font-size: 2.5rem;"></i></h3>
                <h3>No exercises found</h3>
                <p style="font-size: 1.1rem;">Try different keywords like "balance", "beginner", "stress relief", or "hormonal balance"</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 2.5rem;">
            <h3><i class="fa-solid fa-search icon-primary" style="font-size: 3rem;"></i></h3>
            <h2>Search for PCOS/PCOD Exercises</h2>
            <p style="font-size: 1.1rem;">Type keywords like "balance", "beginner", "stress relief", or "hormonal balance" to find exercises!</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: ANALYZE VIDEO
# ==========================================
with tab2:
    st.markdown('<h2 class="result-header"><i class="fa-solid fa-video icon-primary"></i> Upload & Analyze Your Exercise Video</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3 style="margin-top: 0;"><i class="fa-solid fa-lightbulb icon-primary"></i> How it works:</h3>
    <ol style="font-size: 1rem; line-height: 2;">
        <li><b><i class="fa-solid fa-hand-pointer icon-primary"></i> Choose</b> the exercise you're performing from the dropdown</li>
        <li><b><i class="fa-solid fa-upload icon-primary"></i> Upload</b> your video (MP4, MOV, AVI format)</li>
        <li><b><i class="fa-solid fa-robot icon-primary"></i> Analyze</b> - Our AI will detect your pose in real-time</li>
        <li><b><i class="fa-solid fa-download icon-primary"></i> Download</b> the annotated video with visual feedback!</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3><i class="fa-solid fa-dumbbell icon-primary"></i> Select Target Exercise</h3>', unsafe_allow_html=True)
        
        exercise_mapping = {
            "Downdog": "Downdog",
            "Plank": "Plank Pose",
            "Warrior2": "Warrior 2",
            "Modified_Tree": "Modified Tree Pose",
            "Standard_Tree": "Standard Tree Pose"
        }
        
        display_names = list(exercise_mapping.values())
        selected_display = st.selectbox("Choose your exercise:", display_names, key="exercise_select")
        
        reverse_mapping = {v: k for k, v in exercise_mapping.items()}
        target_pose = reverse_mapping[selected_display]
        
        st.markdown(f"""
        <div class="success-box">
        <h3 style="margin: 0;"><i class="fa-solid fa-check-circle icon-primary"></i> Target Exercise Selected</h3>
        <h2 style="margin: 1rem 0 0 0; color: #E0E786; font-size: 1.8rem;">{selected_display}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3><i class="fa-solid fa-upload icon-primary"></i> Upload Your Video</h3>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi'],
            help="Upload a video showing your full body performing the exercise"
        )
    
    if uploaded_video is not None:
        st.markdown("---")
        
        col_preview, col_analyze = st.columns([1, 1])
        
        with col_preview:
            st.markdown('<h3><i class="fa-solid fa-film icon-primary"></i> Your Uploaded Video</h3>', unsafe_allow_html=True)
            video_bytes = uploaded_video.read()
            video_base64 = base64.b64encode(video_bytes).decode()
            st.markdown(f"""
            <div class="video-container">
               <video controls autoplay muted loop>
                  <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
               </video>
             </div>
             """, unsafe_allow_html=True)
             uploaded_video.seek(0)  # Reset file pointer for later use
           
        
        with col_analyze:
            st.markdown('<h3><i class="fa-solid fa-robot icon-primary"></i> Ready to Analyze!</h3>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
            <h3 style="margin-top: 0;"><i class="fa-solid fa-chart-line icon-primary"></i> Analysis Details</h3>
            <p style="font-size: 1rem;"><b><i class="fa-solid fa-bullseye icon-primary"></i> Target Exercise:</b> {selected_display}</p>
            <p style="font-size: 1rem;"><b><i class="fa-solid fa-brain icon-primary"></i> AI Model:</b> Custom Pose Classifier</p>
            <p style="font-size: 1rem;"><b><i class="fa-solid fa-chart-line icon-primary"></i> Accuracy:</b> 92% on validation set</p>
            <p style="font-size: 1rem;"><b><i class="fa-solid fa-gauge-high icon-primary"></i> Processing:</b> Real-time frame analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start AI Analysis", type="primary", use_container_width=True):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                
                with st.spinner("AI is analyzing your video... Please wait!"):
                    results = analyze_video(tfile.name, target_pose)
                
                os.unlink(tfile.name)
                
                if results:
                    st.session_state.analyzed_video_path = results['output_path']
                    
                    st.session_state.exercise_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'target_pose': selected_display,
                        'detected_pose': exercise_mapping.get(results['detected_pose'], results['detected_pose']),
                        'accuracy': results['accuracy'],
                        'confidence': results['confidence']
                    })
                    
                    if len(st.session_state.exercise_history) > 50:
                        st.session_state.exercise_history = st.session_state.exercise_history[-50:]
                    
                    st.markdown("---")
                    st.markdown('<h2 class="result-header"><i class="fa-solid fa-chart-simple icon-primary"></i> Analysis Results</h2>', unsafe_allow_html=True)
                    
                    if results['match']:
                        st.markdown("""
                        <div class="result-status result-perfect">
                        <h2 style="margin: 0; font-size: 2.2rem;">
                            <i class="fa-solid fa-check-circle icon-primary" style="font-size: 2.5rem;"></i> PERFECT MATCH!
                        </h2>
                        <p style="margin: 1.5rem 0 0 0; font-size: 1.2rem; line-height: 1.8;">
                        Excellent work! Your pose matches the target exercise perfectly. Keep up the great form!
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-status result-mismatch">
                        <h2 style="margin: 0; font-size: 2.2rem;"><i class="fa-solid fa-triangle-exclamation icon-accent" style="font-size: 2.5rem;"></i> Different Pose Detected</h2>
                        <p style="margin: 1.5rem 0 0 0; font-size: 1.1rem; line-height: 1.8;">
                        <b><i class="fa-solid fa-bullseye icon-primary"></i> Target Exercise:</b> {selected_display}<br>
                        <b><i class="fa-solid fa-eye icon-accent"></i> Detected Exercise:</b> {exercise_mapping.get(results['detected_pose'], results['detected_pose'])}<br><br>
                        Don't worry! Check the annotated video below to see where adjustments are needed.
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3><i class="fa-solid fa-bullseye icon-primary"></i> Accuracy</h3>
                        <h1>{results['accuracy']:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3><i class="fa-solid fa-brain icon-primary"></i> Confidence</h3>
                        <h1>{results['confidence']*100:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3><i class="fa-solid fa-film icon-primary"></i> Frames</h3>
                        <h1>{results['total_frames']}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown('<h3 style="text-align: center;"><i class="fa-solid fa-video icon-primary"></i> Annotated Video with AI Feedback</h3>', unsafe_allow_html=True)
                    st.markdown('<p style="text-align: center; font-size: 1.1rem; margin-bottom: 1.5rem;"><span style="color: #4CAF50; font-weight: 700;">■ Green</span> = Correct Pose | <span style="color: #f44336; font-weight: 700;">■ Red</span> = Incorrect Pose</p>', unsafe_allow_html=True)
                    
                    with open(results['output_path'], 'rb') as f:
                        video_bytes = f.read()
                        video_base64 = base64.b64encode(video_bytes).decode()
                    st.markdown(f"""
                    <div class="video-container">
                       <video controls autoplay muted loop>
                         <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                        </video>
                    </div>
                     """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    col_action1, col_action2 = st.columns(2)
                    
                    with col_action1:
                        with open(results['output_path'], 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.download_button(
                                label=" Download Annotated Video",
                                data=video_bytes,
                                file_name=f"flexifit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                    
                    with col_action2:
                        if st.button(" Analyze Another Video", use_container_width=True):
                            st.session_state.analyzed_video_path = None
                            st.rerun()

# ==========================================
# TAB 3: AI CHAT
# ==========================================
with tab3:
    st.markdown('<h2 class="result-header"><i class="fa-solid fa-comments icon-primary"></i> Chat with Your AI Exercise Coach</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3 style="margin-top: 0;"><i class="fa-solid fa-question-circle icon-primary"></i> Ask me anything about PCOS/PCOD exercises!</h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem;">
        <div>
            <b style="font-size: 1.1rem;"><i class="fa-solid fa-dumbbell icon-primary"></i> Exercise Questions:</b>
            <ul style="margin: 1rem 0; font-size: 1rem;">
                <li>"What exercises help with PCOS?"</li>
                <li>"How to improve my plank form?"</li>
                <li>"Best poses for stress relief?"</li>
            </ul>
        </div>
        <div>
            <b style="font-size: 1.1rem;"><i class="fa-solid fa-heart-pulse icon-accent"></i> Health & Wellness:</b>
            <ul style="margin: 1rem 0; font-size: 1rem;">
                <li>"Benefits of Tree Pose?"</li>
                <li>"How often should I exercise?"</li>
                <li>"Tips for better hormonal balance?"</li>
            </ul>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #b8c5d6;">
                <h2><i class="fa-solid fa-robot icon-primary" style="font-size: 4rem;"></i></h2>
                <h2 style="color: #E0E786; margin-top: 1rem;">Welcome to AI Coach Chat!</h2>
                <p style="font-size: 1.2rem; margin-top: 1rem;">Start a conversation by typing your question below.</p>
            </div>
            """, unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-user">
                <b style="color: #262626; font-size: 1.1rem;"><i class="fa-solid fa-user icon-primary"></i> You:</b><br>
                <p style="margin: 0.8rem 0 0 0; font-size: 1.05rem; color: #262626; line-height: 1.7;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-assistant">
                <b style="color: #262626; font-size: 1.1rem;"><i class="fa-solid fa-robot" style="color:#919c08;"></i> AI Coach:</b><br>
                <p style="margin: 0.8rem 0 0 0; font-size: 1.05rem; color:#262626; line-height: 1.7;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    user_input = st.text_input(" Type your message...", key="chat_input", placeholder="Ask me anything about PCOS exercises, nutrition, or wellness...")
    
    col_send, col_clear = st.columns([3, 1])
    
    with col_send:
        if st.button("Send Message", use_container_width=True, type="primary"):
            if user_input:
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                if len(st.session_state.chat_history) > 100:
                    st.session_state.chat_history = st.session_state.chat_history[-100:]
                
                with st.spinner(" AI Coach is thinking..."):
                    response = chat_with_ai(user_input)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
            else:
                st.warning(" Please type a message first!")
    
    with col_clear:
        if st.button(" Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ==========================================
# TAB 4: PROGRESS HISTORY
# ==========================================
with tab4:
    st.markdown('<h2 class="result-header"><i class="fa-solid fa-chart-line icon-primary"></i> Your Progress & History</h2>', unsafe_allow_html=True)
    
    history_tab1, history_tab2 = st.tabs(["Exercise Analytics", "Chat History"])
    
    with history_tab1:
        if st.session_state.exercise_history:
            st.markdown(f"<h3 style='text-align: center; margin: 2rem 0;'><i class='fa-solid fa-dumbbell icon-primary'></i> Total Workouts Completed: {len(st.session_state.exercise_history)}</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_acc = np.mean([h['accuracy'] for h in st.session_state.exercise_history])
                st.markdown(f"""
                <div class="metric-card">
                <h3><i class="fa-solid fa-bullseye icon-primary"></i> Avg Accuracy</h3>
                <h1>{avg_acc:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_conf = np.mean([h['confidence'] for h in st.session_state.exercise_history])
                st.markdown(f"""
                <div class="metric-card">
                <h3><i class="fa-solid fa-brain icon-primary"></i> Avg Confidence</h3>
                <h1>{avg_conf*100:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                matches = sum(1 for h in st.session_state.exercise_history if h['target_pose'] == h['detected_pose'])
                success_rate = (matches / len(st.session_state.exercise_history)) * 100
                st.markdown(f"""
                <div class="metric-card">
                <h3><i class="fa-solid fa-trophy icon-primary"></i> Success Rate</h3>
                <h1>{success_rate:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<h3 style="text-align: center;"><i class="fa-solid fa-file-lines icon-primary"></i> Workout History</h3>', unsafe_allow_html=True)
            
            for idx, record in enumerate(reversed(st.session_state.exercise_history)):
                match_status = record['target_pose'] == record['detected_pose']
                
                status_icon = '<i class="fa-solid fa-check-circle" style="color: #C7D06C;"></i>' if match_status else '<i class="fa-solid fa-circle-xmark" style="color: #ff6b6b;"></i>'
                
                with st.expander(f"{status_icon} {record['timestamp']} - {record['target_pose']}", expanded=(idx==0)):
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**<i class='fa-solid fa-bullseye icon-primary'></i> Target Exercise**")
                        st.markdown(f"<h3 style='color: #E0E786;'>{record['target_pose']}</h3>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**<i class='fa-solid fa-eye icon-primary'></i> Detected Exercise**")
                        st.markdown(f"<h3 style='color: #E0E786;'>{record['detected_pose']}</h3>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("**<i class='fa-solid fa-chart-bar icon-primary'></i> Result**")
                        if match_status:
                            st.markdown(f'<h3 style="color: #C7D06C;">{status_icon} Perfect Match</h3>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<h3 style="color: #ff6b6b;">{status_icon} Different Pose</h3>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    progress_col1, progress_col2 = st.columns(2)
                    
                    with progress_col1:
                        st.markdown("**<i class='fa-solid fa-bullseye icon-primary'></i> Accuracy Score**")
                        st.progress(record['accuracy'] / 100)
                        st.caption(f"{record['accuracy']:.1f}%")
                    
                    with progress_col2:
                        st.markdown("**<i class='fa-solid fa-brain icon-primary'></i> AI Confidence**")
                        st.progress(record['confidence'])
                        st.caption(f"{record['confidence']*100:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 3rem;">
                <h2><i class="fa-solid fa-chart-bar icon-primary" style="font-size: 4rem;"></i></h2>
                <h2 style="color: #E0E786; margin-top: 1rem;">No Exercise History Yet</h2>
                <p style="font-size: 1.2rem; margin: 1.5rem 0;">
                Upload and analyze a video in the "Analyze Video" tab to start tracking your progress!
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with history_tab2:
        if st.session_state.chat_history:
            st.markdown(f"<h3 style='text-align: center; margin: 2rem 0;'><i class='fa-solid fa-message icon-primary'></i> Total Conversations: {len(st.session_state.chat_history) // 2}</h3>", unsafe_allow_html=True)
            
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-user">
                    <b style="color: #E0E786; font-size: 1.1rem;"><i class="fa-solid fa-user icon-primary"></i> You:</b><br>
                    <p style="margin: 0.8rem 0 0 0; font-size: 1.05rem; line-height: 1.7;">{message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-assistant">
                    <b style="color: #b8c5d6; font-size: 1.1rem;"><i class="fa-solid fa-robot"></i> AI Coach:</b><br>
                    <p style="margin: 0.8rem 0 0 0; font-size: 1.05rem; line-height: 1.7;">{message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 3rem;">
                <h2><i class="fa-solid fa-comment icon-primary" style="font-size: 4rem;"></i></h2>
                <h2 style="color: #E0E786; margin-top: 1rem;">No Chat History Yet</h2>
                <p style="font-size: 1.2rem; margin: 1.5rem 0;">
                Start a conversation with the AI Coach in the "AI Coach Chat" tab!
                </p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
    """, unsafe_allow_html=True)
    
    st.image("logo.png", width=100)
    
    st.markdown("""
        <h2 style="color: #919c08; margin: 1rem 0; font-weight: 900; font-size: 1.8rem;">FLEXIFIT AI</h2>
        <p style="color: #262626; margin: 0; font-size: 1rem;">PCOS/PCOD Exercise Coach</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="color: #919c08;">
        <h3 style="color: #919c08; margin-bottom: 1.5rem;"><i class="fa-solid fa-bolt icon-primary"></i> Powered By</h3>
        <ul style="list-style: none; padding: 0;">
            <li style="padding: 0.8rem 0;">
                <b style="font-size: 1.05rem;"><i class="fa-solid fa-person-running icon-primary"></i> MediaPipe</b><br>
                <span style="opacity: 0.8; color:#262626; font-size: 0.95rem;">Real-time pose detection</span>
            </li>
            <li style="padding: 0.8rem 0;">
                <b style="font-size: 1.05rem;"><i class="fa-solid fa-brain icon-primary"></i> Custom ML Model</b><br>
                <span style="opacity: 0.8; color: #262626; font-size: 0.95rem;">92% accuracy classification</span>
            </li>
            <li style="padding: 0.8rem 0;">
                <b style="font-size: 1.05rem;"><i class="fa-solid fa-database icon-primary"></i> Elasticsearch</b><br>
                <span style="opacity: 0.8; color: #262626; font-size: 0.95rem;">Smart exercise search</span>
            </li>
            <li style="padding: 0.8rem 0;">
                <b style="font-size: 1.05rem;"><i class="fa-solid fa-robot icon-primary"></i> Vertex AI Gemini</b><br>
                <span style="opacity: 0.8; color: #262626; font-size: 0.95rem;">Intelligent coaching</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button(" Clear All History", use_container_width=True):
        st.session_state.exercise_history = []
        st.session_state.chat_history = []
        st.success("All history cleared!")
        st.rerun()





















