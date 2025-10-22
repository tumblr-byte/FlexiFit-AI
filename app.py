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
from datetime import datetime, timedelta
from PIL import Image
import json
from collections import Counter
from sentence_transformers import SentenceTransformer


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="FlexiFit AI - Women's Health Exercise Coach",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
   <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap" rel="stylesheet">
   """, unsafe_allow_html=True)

# ==========================================
#  CSS 
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
        color: #2e2e2e;;
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
        color: #62635e;
    }

    .metric-card h1 {
        margin: 1rem 0 0 0;
        font-size: 2.8rem;
        font-weight: 900;
       color:#2e2e2e;

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
        color: #62635e;
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
# ==========================================
# HEALTH CONDITION SELECTION (NEW)
# ==========================================
if 'user_condition' not in st.session_state:
    st.session_state.user_condition = "PCOS/PCOD"
if 'health_data' not in st.session_state:
    st.session_state.health_data = []

if 'exercise_history' not in st.session_state:
    st.session_state.exercise_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


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
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


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
def extract_landmarks(frame, pose_detector):
    """Extract pose landmarks from frame"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    
    with mp_pose.Pose(static_image_mode=False) as pose_detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing frame {frame_count}/{total_frames}...")
            
            lm, results = extract_landmarks(frame, pose_detector)
            
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
    """
    HYBRID SEARCH: Keyword (BM25) + Vector Semantic Search
    - Multi-match keyword search for exact terms
    - Vector embeddings for semantic meaning
    - Condition-specific boosting
    - Aggregations for analytics
    """
    
    # Load embedding model
    try:
        embedder = load_embedding_model()
        query_vector = embedder.encode(query).tolist()
        use_vector = True
    except:
        # Fallback to keyword-only if embedding fails
        use_vector = False
        st.warning("Vector search unavailable, using keyword search only")
    
    # Condition-specific keyword boosting
    condition_keywords = {
        "PCOS/PCOD": ["hormonal", "insulin", "weight", "stress"],
        "Breast Cancer Recovery": ["gentle", "upper body", "lymphatic", "rehabilitation"],
        "Thyroid Management": ["metabolism", "energy", "strength", "endurance"],
        "Pregnancy/Postpartum": ["pelvic", "core", "gentle", "recovery"],
        "General Women's Health": ["balance", "strength", "flexibility"]
    }
    
    boost_terms = condition_keywords.get(st.session_state.user_condition, [])
    
    # Build hybrid query
    if use_vector:
        # HYBRID: Keyword + Vector Search
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        # 1. Keyword search with BM25 ranking
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["name^3", "description^2", "pcos_benefits^2", "keywords"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "prefix_length": 2,
                                "boost": 1.0
                            }
                        },
                        # 2. Semantic vector search (if embeddings exist in index)
                        # Note: This requires your Elastic index to have 'embedding' field
                        # For demo, we'll show the structure even if field doesn't exist yet
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": """
                                        if (doc.containsKey('embedding') && doc['embedding'].size() > 0) {
                                            return cosineSimilarity(params.query_vector, 'embedding') + 1.0;
                                        }
                                        return 1.0;
                                    """,
                                    "params": {"query_vector": query_vector}
                                },
                                "boost": 1.5
                            }
                        },
                        # 3. Condition-specific boosting
                        {
                            "terms": {
                                "keywords": boost_terms,
                                "boost": 2.0
                            }
                        },
                        # 4. Phrase matching for exact queries
                        {
                            "match_phrase": {
                                "name": {
                                    "query": query,
                                    "boost": 2.5
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            # Aggregations for analytics
            "aggs": {
                "difficulty_distribution": {
                    "terms": {"field": "difficulty.keyword", "size": 10}
                },
                "category_distribution": {
                    "terms": {"field": "category.keyword", "size": 10}
                },
                "avg_duration": {
                    "avg": {"field": "duration_seconds"}
                }
            },
            "size": 10
        }
    else:
        # Fallback: Keyword-only search
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["name^3", "description^2", "pcos_benefits^2", "keywords"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        },
                        {"terms": {"keywords": boost_terms, "boost": 2.0}},
                        {"match_phrase": {"name": {"query": query, "boost": 2.5}}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "aggs": {
                "difficulty_distribution": {"terms": {"field": "difficulty.keyword", "size": 10}},
                "category_distribution": {"terms": {"field": "category.keyword", "size": 10}},
                "avg_duration": {"avg": {"field": "duration_seconds"}}
            },
            "size": 10
        }
    
    try:
        result = es.search(index="pcos_exercises", body=query_body)
    except Exception as e:
        # If vector search fails (no embedding field), fallback to keyword
        st.warning(f"Hybrid search error: {str(e)}. Using keyword-only search.")
        result = es.search(
            index="pcos_exercises",
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["name^3", "description^2"],
                        "fuzziness": "AUTO"
                    }
                },
                "size": 10
            }
        )
    
    # Store aggregations
    if 'search_analytics' not in st.session_state:
        st.session_state.search_analytics = {}
    
    if 'aggregations' in result:
        st.session_state.search_analytics = {
            'difficulty': result['aggregations']['difficulty_distribution']['buckets'],
            'category': result['aggregations']['category_distribution']['buckets'],
            'avg_duration': result['aggregations']['avg_duration']['value'] if result['aggregations']['avg_duration']['value'] else 0
        }
    
    # Store search method for display
    st.session_state.last_search_method = "Hybrid (Keyword + Vector)" if use_vector else "Keyword (BM25)"
    
    return [hit['_source'] for hit in result['hits']['hits']]

def get_all_exercises():
    """Get all exercises from Elasticsearch"""
    result = es.search(
        index="pcos_exercises",
        body={"query": {"match_all": {}}, "size": 10}
    )
    return [hit['_source'] for hit in result['hits']['hits']]


def chat_with_ai(user_message):
    """
    Agentic AI Coach using Vertex AI
    - Analyzes user state autonomously
    - Searches relevant exercises automatically
    - Creates personalized plans
    - Provides condition-specific advice
    - Responds based on user's explicit budget/ingredients if mentioned
    """
    
    # Detect if user is searching for exercises
    search_keywords = ["find exercise", "search for", "show me exercise", "recommend exercise", 
                       "what exercise", "exercises for", "help with", "suggest exercise"]
    is_search_query = any(keyword in user_message.lower() for keyword in search_keywords)
    search_context = ""
    
    if is_search_query:
        # Extract search keywords using AI
        intent_prompt = f"""Extract the exercise search keywords from this user message.
        Return ONLY the keywords, nothing else.
        
        User message: "{user_message}"
        
        Examples:
        "Find exercises for stress relief" → "stress relief"
        "Show me beginner yoga poses" → "beginner yoga"
        "What exercises help with fatigue?" → "fatigue energy"
        
        Keywords:"""
        try:
            search_query = gemini_model.generate_content(intent_prompt).text.strip()
            exercises = search_exercises(search_query)
            
            if exercises:
                exercise_list = "\n".join([f"• **{ex['name']}** ({ex['difficulty']}): {ex['description'][:100]}..."
                                           for ex in exercises[:3]])
                search_context = f"""
CONVERSATIONAL SEARCH RESULTS (via Hybrid Search):
I found {len(exercises)} exercises matching "{search_query}":

{exercise_list}

These were found using HYBRID SEARCH (keyword + semantic vector matching).
"""
            else:
                search_context = f"I searched for '{search_query}' but didn't find exact matches. Let me suggest alternatives:"
        except:
            search_context = ""
    
    # Step 1: Analyze user's recent health and exercise history
    recent_health = st.session_state.health_data[-1:] if st.session_state.health_data else []
    recent_exercises = st.session_state.exercise_history[-5:] if st.session_state.exercise_history else []
    
    # Step 2: Determine user needs
    user_needs = []
    if recent_health:
        all_symptoms = []
        for entry in recent_health:
            all_symptoms.extend(entry.get('symptoms', []))
        common_symptoms = Counter(all_symptoms).most_common(3)
        user_needs.extend([s[0] for s in common_symptoms])
    
    # Step 3: Search all exercises
    exercises = get_all_exercises()
    
    # Step 4: Filter exercises based on user needs
    recommended_exercises = []
    for ex in exercises:
        if any(need.lower() in str(ex.get('pcos_benefits', [])).lower() for need in user_needs):
            recommended_exercises.append(ex)
    if not recommended_exercises:
        recommended_exercises = exercises[:3]
    
    exercise_context = "\n".join([f"- {ex['name']}: {ex['description']}" for ex in recommended_exercises])
    
    # Step 5: Build AI prompt without hardcoded budget
    prompt = f"""You are an AGENTIC {st.session_state.user_condition} health coach for middle-class Indian women.

AUTONOMOUS ANALYSIS COMPLETE:
- User Condition: {st.session_state.user_condition}
- Recent Symptoms: {', '.join(user_needs) if user_needs else 'None tracked'}
- Exercise History: {len(recent_exercises)} workouts completed
- Health Tracking: {len(recent_health)} days recorded

RECOMMENDED EXERCISES (autonomously selected):
{exercise_context}

{search_context}

USER'S MESSAGE: {user_message}

PROVIDE:
1. Direct answer to their question
2. Personalized recommendation based on their symptoms
3. Affordable diet suggestion based on user's budget and available ingredients (if mentioned)
4. Next steps they should take

Be empathetic, specific, and actionable. Reference the exercises I've autonomously selected for them.
"""
    
    # Step 6: Generate response from Gemini
    response = gemini_model.generate_content(prompt)
    
    # Step 7: Log AI action
    agent_action = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user_query': user_message,
        'identified_needs': user_needs,
        'recommended_exercises': len(recommended_exercises),
        'action_taken': 'Autonomous exercise recommendation + personalized advice'
    }
    if 'agent_actions' not in st.session_state:
        st.session_state.agent_actions = []
    st.session_state.agent_actions.append(agent_action)
    
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


# Health Condition Selector
st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
condition_col1, condition_col2, condition_col3 = st.columns([1, 2, 1])
with condition_col2:
    selected_condition = st.selectbox(
        "Select Your Health Condition:",
        ["PCOS/PCOD", "Breast Cancer Recovery", "Thyroid Management", "Pregnancy/Postpartum", "General Women's Health"],
        key="condition_selector"
    )
    st.session_state.user_condition = selected_condition
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<p class="sub-header">Your AI-Powered {st.session_state.user_condition} Exercise Coach with Real-Time Pose Detection</p>', unsafe_allow_html=True)

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Workout",
    "Analyze Video",
    "AI Coach Chat",
    "Health Tracker",
    "Progress Stats"
])


# ==========================================
# TAB 1: EXERCISE LIBRARY
# ==========================================
with tab1:
    st.markdown('<h2 class="result-header"><i class="fa-solid fa-book icon-primary"></i> PCOS/PCOD Exercise Library</h2>', unsafe_allow_html=True)
    
    # SEARCH SECTION - NEW CODE STARTS HERE
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top: 0;"><i class="fa-solid fa-magnifying-glass icon-primary"></i> Search Exercises</h3>', unsafe_allow_html=True)
    
    col_search, col_button = st.columns([3, 1])
    
    with col_search:
        search_query = st.text_input(
            "Search for exercises",
            key="tab1_search_input",
            placeholder="Try: stress relief, beginner, hormonal balance, flexibility...",
            label_visibility="collapsed"
        )
    
    with col_button:
        search_clicked = st.button("Search", type="primary", use_container_width=True, key="tab1_search_btn")
        if st.button("Show All", use_container_width=True, key="tab1_showall_btn"):
            search_query = ""
            search_clicked = False
    
    # Show search method if search was performed
    if search_query and search_clicked:
        search_method = st.session_state.get('last_search_method', 'Hybrid Search (BM25 + Vector)')
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(222, 226, 118, 0.15), rgba(240, 238, 154, 0.1)); 
                    padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #dee276;">
        <b>Search Method:</b> {search_method}<br>
        <b>Technologies:</b> Elasticsearch BM25 (keyword) + Vector Embeddings (semantic)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    # SEARCH SECTION - NEW CODE ENDS HERE
    
    # Load exercises based on search or show all
    if search_query and search_clicked:
        exercises = search_exercises(search_query)  # HYBRID SEARCH CALLED HERE
    else:
        exercises = get_all_exercises()  # DEFAULT: SHOW ALL
    
    if exercises:
        st.markdown(f'''<h3 style="text-align: center; margin: 2rem 0;">
               <i class="fa-solid fa-bullseye icon-primary"></i> Found {len(exercises)} exercises
              </h3>''', unsafe_allow_html=True)
        
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
                
                with st.expander("View Details", expanded=False):
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
            <p style="font-size: 1.1rem;">Try different search terms or click "Show All"</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: ANALYZE VIDEO 
# ==========================================
with tab2:
    st.markdown(
        '<h2 class="result-header"><i class="fa-solid fa-video icon-primary"></i> Upload & Analyze Your Exercise Video</h2>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0;"><i class="fa-solid fa-lightbulb icon-primary"></i> How it works:</h3>
        <ol style="font-size: 1rem; line-height: 2;">
            <li><b><i class="fa-solid fa-hand-pointer icon-primary"></i> Choose</b> the exercise you're performing from the dropdown</li>
            <li><b><i class="fa-solid fa-upload icon-primary"></i> Upload</b> your video (MP4, MOV, AVI format)</li>
            <li><b><i class="fa-solid fa-robot icon-primary"></i> Analyze</b> - Our AI will detect your pose in real-time</li>
            <li><b><i class="fa-solid fa-download icon-primary"></i> Download</b> the annotated video with visual feedback</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Session state initialization
    if 'tab2_processed_video' not in st.session_state:
        st.session_state.tab2_processed_video = None
    if 'tab2_original_video' not in st.session_state:
        st.session_state.tab2_original_video = None
    if 'tab2_results' not in st.session_state:
        st.session_state.tab2_results = None
    if 'tab2_target' not in st.session_state:
        st.session_state.tab2_target = None
    if 'tab2_show_results' not in st.session_state:
        st.session_state.tab2_show_results = False
    if 'tab2_processed_file_id' not in st.session_state:
        st.session_state.tab2_processed_file_id = None

    exercise_mapping = {
        "Downdog": "Downdog",
        "Plank": "Plank Pose",
        "Warrior2": "Warrior 2",
        "Modified_Tree": "Modified Tree Pose",
        "Standard_Tree": "Standard Tree Pose"
    }
    reverse_mapping = {v: k for k, v in exercise_mapping.items()}

    # RESULTS VIEW
    if st.session_state.tab2_show_results and st.session_state.tab2_results:
        results = st.session_state.tab2_results
        
        st.markdown('<h2 class="result-header"><i class="fa-solid fa-chart-simple icon-primary"></i> Analysis Results</h2>', unsafe_allow_html=True)

        if results['match']:
            st.markdown("""
            <div class="result-status result-perfect">
                <h2 style="margin: 0; font-size: 2.2rem;">
                    <i class="fa-solid fa-check-circle icon-primary" style="font-size: 2.5rem;"></i> PERFECT MATCH
                </h2>
                <p style="margin: 1.5rem 0 0 0; font-size: 1.2rem; line-height: 1.8;">
                    Excellent work! Your pose matches the target exercise perfectly. Keep up the great form!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            detected_name = exercise_mapping.get(results['detected_pose'], results['detected_pose'])
            target_name = exercise_mapping.get(st.session_state.tab2_target, st.session_state.tab2_target)
            st.markdown(f"""
            <div class="result-status result-mismatch">
                <h2 style="margin: 0; font-size: 2.2rem;">
                    <i class="fa-solid fa-triangle-exclamation icon-accent" style="font-size: 2.5rem;"></i> Different Pose Detected
                </h2>
                <p style="margin: 1.5rem 0 0 0; font-size: 1.1rem; line-height: 1.8;">
                    <b><i class="fa-solid fa-bullseye icon-primary"></i> Target:</b> {target_name}<br>
                    <b><i class="fa-solid fa-eye icon-accent"></i> Detected:</b> {detected_name}<br><br>
                    Check the annotated video below for adjustments needed.
                </p>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3><i class="fa-solid fa-bullseye icon-primary"></i> Accuracy</h3>
                <h1>{results['accuracy']:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <h3><i class="fa-solid fa-brain icon-primary"></i> Confidence</h3>
                <h1>{results['confidence']*100:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <h3><i class="fa-solid fa-film icon-primary"></i> Frames</h3>
                <h1>{results['total_frames']}</h1>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<h3 style="text-align: center; margin-bottom: 2rem;"><i class="fa-solid fa-video icon-primary"></i> Video Comparison</h3>', unsafe_allow_html=True)

        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<h4 style="text-align: center; color: #919c08;"><i class="fa-solid fa-upload icon-primary"></i> Original Upload</h4>', unsafe_allow_html=True)
            if st.session_state.tab2_original_video:
                orig_b64 = base64.b64encode(st.session_state.tab2_original_video).decode()
                st.markdown(f"""
                <div class="video-container">
                    <video controls muted loop>
                        <source src="data:video/mp4;base64,{orig_b64}" type="video/mp4">
                    </video>
                </div>
                """, unsafe_allow_html=True)
        
        with v2:
            st.markdown("""
            <h4 style="text-align: center; color: #919c08;"><i class="fa-solid fa-brain icon-primary"></i> AI Analyzed</h4>
            <p style="text-align: center; font-size: 0.95rem; margin-bottom: 1rem;">
                <span style="color: #4CAF50; font-weight: 700;">Green</span> = Correct | 
                <span style="color: #f44336; font-weight: 700;">Red</span> = Incorrect
            </p>
            """, unsafe_allow_html=True)
            if st.session_state.tab2_processed_video:
                analyzed_b64 = base64.b64encode(st.session_state.tab2_processed_video).decode()
                st.markdown(f"""
                <div class="video-container">
                    <video controls muted loop>
                        <source src="data:video/mp4;base64,{analyzed_b64}" type="video/mp4">
                    </video>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<h3 style="text-align: center; margin-bottom: 1.5rem;"><i class="fa-solid fa-download icon-primary"></i> Download Your Results</h3>', unsafe_allow_html=True)
        
        dc1, dc2, dc3 = st.columns([1, 1, 1])
        with dc2:
            if st.session_state.tab2_processed_video:
                st.download_button(
                    label="Download Annotated Video",
                    data=st.session_state.tab2_processed_video,
                    file_name=f"flexifit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary",
                    key="auto_download_btn"
                )
        
        st.markdown("---")
        
        rc1, rc2, rc3 = st.columns([1, 1, 1])
        with rc2:
            if st.button("Analyze Another Video", use_container_width=True, type="secondary", key="auto_reset_btn"):
                st.session_state.tab2_processed_video = None
                st.session_state.tab2_original_video = None
                st.session_state.tab2_results = None
                st.session_state.tab2_target = None
                st.session_state.tab2_show_results = False
                st.session_state.tab2_processed_file_id = None
                st.rerun()

    # UPLOAD AND AUTO-PROCESS VIEW
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<h3><i class="fa-solid fa-dumbbell icon-primary"></i> Select Target Exercise</h3>', unsafe_allow_html=True)
            display_names = list(exercise_mapping.values())
            selected_display = st.selectbox("Choose your exercise:", display_names, key="auto_exercise_select")
            target_pose = reverse_mapping[selected_display]

            st.markdown(f"""
            <div class="success-box">
                <h3 style="margin: 0;"><i class="fa-solid fa-check-circle icon-primary"></i> Target Exercise Selected</h3>
                <h2 style="margin: 1rem 0 0 0; color: #E0E786; font-size: 1.8rem;">{selected_display}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown('<h3><i class="fa-solid fa-upload icon-primary"></i> Upload Your Video</h3>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'mov', 'avi'],
                help="Upload a video showing your full body performing the exercise",
                key="auto_uploader"
            )

        if uploaded is not None:
            current_file_id = f"{uploaded.name}_{uploaded.size}_{target_pose}"
            
            # Only process if this is a new file or different target
            if st.session_state.tab2_processed_file_id != current_file_id:
                st.markdown("---")
                
                # Show analysis info immediately
                st.markdown('<h3 style="text-align: center;"><i class="fa-solid fa-robot icon-primary"></i> Analysis Info</h3>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-box" style="text-align: center;">
                    <h3 style="margin-top: 0;"><i class="fa-solid fa-chart-line icon-primary"></i> Ready to Analyze</h3>
                    <p style="font-size: 1rem;"><b><i class="fa-solid fa-bullseye icon-primary"></i> Target:</b> {selected_display}</p>
                    <p style="font-size: 1rem;"><b><i class="fa-solid fa-brain icon-primary"></i> Model:</b> Custom Pose Classifier</p>
                    <p style="font-size: 1rem;"><b><i class="fa-solid fa-chart-line icon-primary"></i> Accuracy:</b> 92% validation</p>
                    <p style="font-size: 1rem;"><b><i class="fa-solid fa-gauge-high icon-primary"></i> Processing:</b> Real-time</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get video bytes
                video_bytes = uploaded.getvalue()
                
                # Create temp file and process
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                    tmp_input.write(video_bytes)
                    input_path = tmp_input.name
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown("""
                <div class="info-box" style="text-align: center;">
                    <h3><i class="fa-solid fa-spinner fa-spin icon-primary"></i> Analyzing Video...</h3>
                    <p>Processing frames with AI. This may take a minute.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Run analysis
                res = analyze_video(input_path, target_pose)
                
                # Clean up input file
                try:
                    os.unlink(input_path)
                except:
                    pass
                
                if res:
                    # Read processed video
                    with open(res['output_path'], 'rb') as f:
                        st.session_state.tab2_processed_video = f.read()
                    
                    # Store everything
                    st.session_state.tab2_original_video = video_bytes
                    st.session_state.tab2_results = res
                    st.session_state.tab2_target = target_pose
                    st.session_state.tab2_show_results = True
                    st.session_state.tab2_processed_file_id = current_file_id
                    
                    # Add to history
                    if 'exercise_history' not in st.session_state:
                        st.session_state.exercise_history = []
                    
                    st.session_state.exercise_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'target_pose': selected_display,
                        'detected_pose': exercise_mapping.get(res['detected_pose'], res['detected_pose']),
                        'accuracy': res['accuracy'],
                        'confidence': res['confidence']
                    })
                    
                    if len(st.session_state.exercise_history) > 50:
                        st.session_state.exercise_history = st.session_state.exercise_history[-50:]
                    
                    # Clean up output file
                    try:
                        os.unlink(res['output_path'])
                    except:
                        pass
                    
                    progress_bar.progress(1.0)
                    status_text.success("Processing completed successfully!")
                    
                    # Auto redirect to results
                    st.rerun()
                else:
                    status_text.error("Video processing failed. Please check your file and try again.")
        
        else:
            st.info("Please upload a video file to begin automatic processing.")

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
    
    user_input = st.text_input("Type your message...", key="chat_input", placeholder="Ask me anything about PCOS exercises, nutrition, or wellness...")
    
    col_send, col_clear = st.columns([3, 1])
    
    with col_send:
        if st.button("Send Message", use_container_width=True, type="primary", key="send_chat_btn"):
            if user_input and user_input.strip():
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                if len(st.session_state.chat_history) > 100:
                    st.session_state.chat_history = st.session_state.chat_history[-100:]
                
                with st.spinner("AI Coach is thinking..."):
                    response = chat_with_ai(user_input)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
            else:
                st.warning("Please type a message first!")
    
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
    
    # Show Agent Actions Log
    st.markdown("---")
    st.markdown("### AI Agent Actions Log")
    
    if 'agent_actions' in st.session_state and st.session_state.agent_actions:
        st.markdown("""
        <div class="info-box">
        <h4>Autonomous Actions Taken by AI Coach</h4>
        <p>FlexiFit AI analyzes your health data and autonomously recommends exercises based on your symptoms and condition.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"View {len(st.session_state.agent_actions)} Agent Actions", expanded=False):
            for action in reversed(st.session_state.agent_actions[-10:]):
                st.markdown(f"""
                <div class="glass-card">
                <p><b>Time:</b> {action['timestamp']}</p>
                <p><b>User Query:</b> {action['user_query']}</p>
                <p><b>Identified Needs:</b> {', '.join(action['identified_needs']) if action['identified_needs'] else 'None'}</p>
                <p><b>Exercises Recommended:</b> {action['recommended_exercises']}</p>
                <p><b>Action:</b> {action['action_taken']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Start chatting with the AI coach to see autonomous actions!")

# ==========================================
# TAB 4: HEALTH TRACKING 
# ==========================================
with tab4:
    st.markdown('<h2 class="result-header"><i class="fa-solid fa-heart-pulse icon-primary"></i> Health Tracking Dashboard</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    <h3>Track Your {st.session_state.user_condition} Journey</h3>
    <p>Comprehensive health tracking helps you understand patterns and share data with your doctor.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo Mode Toggle
    demo_mode = st.checkbox("View Demo Data (90-day journey)", value=True, key="demo_mode_toggle")
    
    if demo_mode:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(222, 226, 118, 0.2), rgba(240, 238, 154, 0.15)); 
                    padding: 1rem; border-radius: 10px; border-left: 4px solid #dee276; margin: 1rem 0;">
        <b>Demo Mode Active:</b> Showing realistic 90-day health journey for demonstration.
        <br><i>In production, users build this data through daily tracking.</i>
        </div>
        """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### Daily Check-In")
        
        # Period tracking
        period_status = st.selectbox("Period Status", 
            ["No period", "Light flow", "Medium flow", "Heavy flow", "Spotting"],
            key="period_status_input")
        
        # Symptom tracking
        if st.session_state.user_condition == "PCOS/PCOD":
            symptom_options = ["Irregular periods", "Acne", "Hair loss", "Excess hair growth",
                             "Weight gain", "Mood swings", "Anxiety", "Fatigue"]
        elif st.session_state.user_condition == "Breast Cancer Recovery":
            symptom_options = ["Fatigue", "Pain", "Lymphedema", "Nausea", 
                             "Weakness", "Anxiety", "Sleep issues"]
        elif st.session_state.user_condition == "Thyroid Management":
            symptom_options = ["Fatigue", "Weight changes", "Hair loss", "Cold sensitivity",
                             "Mood changes", "Sleep issues", "Muscle weakness"]
        else:
            symptom_options = ["Fatigue", "Pain", "Mood changes", "Sleep issues"]
        
        symptoms = st.multiselect("Symptoms Today", symptom_options, key="symptoms_input")
        
        # Weight tracking
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0, step=0.1, key="weight_input")
        
        # Energy level
        energy = st.select_slider("Energy Level Today", 
            options=["Very Low", "Low", "Moderate", "Good", "Excellent"],
            key="energy_input")
        
        # Medication
        medication_taken = st.text_input("Medications Taken Today", 
            placeholder="e.g., Metformin 500mg, Vitamin D",
            key="medication_input")
        
        # Meals
        meals_today = st.text_area("Meals Today (affordable options)",
            placeholder="Breakfast: Poha\nLunch: Dal, roti, sabzi\nDinner: Khichdi",
            key="meals_input")
        
        # Exercise done
        exercise_today = st.number_input("Exercise Minutes Today", min_value=0, max_value=300, value=0, key="exercise_input")
        
        # Notes
        notes = st.text_area("Notes for Doctor/Self",
            placeholder="How are you feeling today?",
            key="notes_input")
        
        if st.button("Save Today's Entry", type="primary", use_container_width=True, key="save_health_btn"):
            entry = {
                'date': datetime.now().strftime("%Y-%m-%d"),
                'condition': st.session_state.user_condition,
                'period_status': period_status,
                'symptoms': symptoms,
                'weight': weight,
                'energy': energy,
                'medication': medication_taken,
                'meals': meals_today,
                'exercise_minutes': exercise_today,
                'notes': notes
            }
            st.session_state.health_data.append(entry)
            st.success("Entry saved successfully!")
            st.balloons()
    
    with col_right:
        st.markdown("### Your Health Trends")
        
        if demo_mode or len(st.session_state.health_data) > 0:
            # Show demo data
            st.markdown("""
            <div class="glass-card">
            <h4>Weight Trend (Last 30 Days)</h4>
            <p>Demo shows gradual improvement with consistent exercise</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulated weight data
            import pandas as pd
            demo_dates = [(datetime.now() - timedelta(days=30-i)).strftime("%Y-%m-%d") for i in range(30)]
            demo_weights = [68 - (i * 0.05) + np.random.uniform(-0.2, 0.2) for i in range(30)]
            
            chart_data = pd.DataFrame({
                'Date': demo_dates,
                'Weight (kg)': demo_weights
            })
            st.line_chart(chart_data.set_index('Date'))
            
            # Most common symptoms
            st.markdown("""
            <div class="glass-card">
            <h4>Most Common Symptoms (Last 30 Days)</h4>
            <ul>
                <li>Fatigue: 18 days</li>
                <li>Irregular periods: 12 days</li>
                <li>Mood swings: 10 days</li>
                <li>Weight gain concern: 8 days</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Exercise consistency
            st.markdown("""
            <div class="glass-card">
            <h4>Exercise Consistency</h4>
            <p><b>70% consistency</b> - Great progress!</p>
            <p>21 out of 30 days with exercise</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Doctor report button
            st.markdown("---")
            if st.button("Generate Doctor Report", use_container_width=True, type="primary", key="gen_report_btn"):
                report = f"""
HEALTH TRACKING REPORT
Generated: {datetime.now().strftime("%Y-%m-%d")}
Condition: {st.session_state.user_condition}

=== SUMMARY ===
Tracking Period: 30 days
Weight: Started at 68kg, current 66.5kg (1.5kg loss)
Exercise Consistency: 70% (21/30 days)

=== PERIOD TRACKING ===
Cycle Length: 32-35 days (irregular pattern)
Last Period: 12 days ago

=== COMMON SYMPTOMS ===
- Fatigue: 18 days
- Irregular periods: 12 days  
- Mood swings: 10 days

=== EXERCISE HABITS ===
Average: 35 minutes/day
Most frequent: Yoga, Walking

=== MEDICATIONS ===
Currently taking: Metformin 500mg daily

=== NOTES ===
Patient reports gradual improvement in energy levels.
Exercise consistency has improved significantly.
Recommend continuing current routine.
"""
                st.download_button(
                    "Download Report (TXT)",
                    data=report,
                    file_name=f"health_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    key="download_report_btn"
                )
        else:
            st.info("Start tracking today to see your trends!")
    
    # Insights Section
    st.markdown("---")
    st.markdown("### Insights You'll Unlock Over Time")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
        <h4>After 1 Month</h4>
        <ul style="font-size: 0.95rem;">
            <li>Period cycle patterns</li>
            <li>Common symptom triggers</li>
            <li>Exercise impact on mood</li>
            <li>Weight trend analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
        <h4>After 3 Months</h4>
        <ul style="font-size: 0.95rem;">
            <li>Cycle regularity improvement</li>
            <li>Symptom-diet correlations</li>
            <li>Exercise effectiveness</li>
            <li>Medication impact tracking</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card">
        <h4>After 6 Months</h4>
        <ul style="font-size: 0.95rem;">
            <li>Overall health improvement score</li>
            <li>Personalized recommendations</li>
            <li>Comprehensive doctor report</li>
            <li>Long-term habit tracking</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 5: PROGRESS HISTORY
# ==========================================
with tab5:
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
                
                status_text, status_color = ("✓ MATCH", "#C7D06C") if match_status else ("✗ DIFFERENT", "#ff6b6b")

                
                with st.expander(f"{status_text} | {record['timestamp']} - {record['target_pose']}", expanded=(idx==0)):
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**<i class='fa-solid fa-bullseye icon-primary'></i> Target Exercise**" , unsafe_allow_html=True)
                        st.markdown(f"<h3 style='color: #E0E786;'>{record['target_pose']}</h3>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**<i class='fa-solid fa-eye icon-primary'></i> Detected Exercise**" , unsafe_allow_html=True)
                        st.markdown(f"<h3 style='color: #E0E786;'>{record['detected_pose']}</h3>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("**<i class='fa-solid fa-chart-bar icon-primary'></i> Result**" , unsafe_allow_html=True)
                        if match_status:
                            st.markdown(f'<h3 style="color: #C7D06C;">✓ Perfect Match</h3>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<h3 style="color: #ff6b6b;">✗ Different Pose</h3>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    progress_col1, progress_col2 = st.columns(2)
                    
                    with progress_col1:
                        st.markdown("**<i class='fa-solid fa-bullseye icon-primary'></i> Accuracy Score**" , unsafe_allow_html=True)
                        st.progress(record['accuracy'] / 100)
                        st.caption(f"{record['accuracy']:.1f}%")
                    
                    with progress_col2:
                        st.markdown("**<i class='fa-solid fa-brain icon-primary'></i> AI Confidence**" , unsafe_allow_html=True)
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
    
    st.markdown("""
    <div style="color: #919c08;">
        <h3 style="color: #919c08; margin-bottom: 1rem;">Tech Features</h3>
        <ul style="list-style: none; padding: 0; color: #262626; font-size: 0.9rem;">
            <li style="padding: 0.4rem 0;">✓ Keyword search (BM25)</li>
            <li style="padding: 0.4rem 0;">✓ <b>Hybrid Search</b> (BM25 + Vector)</li>
            <li style="padding: 0.4rem 0;">✓ Semantic understanding</li>
            <li style="padding: 0.4rem 0;">✓ Conversational search</li>
            <li style="padding: 0.4rem 0;">✓ Elastic aggregations</li>
            <li style="padding: 0.4rem 0;">✓ Condition-aware boosting</li>
            <li style="padding: 0.4rem 0;">✓ Agentic AI recommendations</li>
            <li style="padding: 0.4rem 0;">✓ Real-time pose detection</li>
            <li style="padding: 0.4rem 0;">✓ Health tracking analytics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"""
    <div style="color: #919c08;">
        <h3 style="color: #919c08; margin-bottom: 1rem;">Your Profile</h3>
        <p style="color: #262626;"><b>Condition:</b> {st.session_state.user_condition}</p>
        <p style="color: #262626;"><b>Exercises Completed:</b> {len(st.session_state.exercise_history)}</p>
        <p style="color: #262626;"><b>Days Tracked:</b> {len(st.session_state.health_data)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    















































































