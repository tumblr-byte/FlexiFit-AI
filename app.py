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
# CUSTOM CSS WITH GLASSMORPHISM
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
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0a0a0a 100%);
        padding: 1.5rem 1rem;
    }
    
    /* Large, bold header - no animation */
    .main-header {
        font-size: 5.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #C7D06C 0%, #E0E786 50%, #C7D06C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 4px;
        margin: 1rem 0;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: 0.95rem;
        text-align: center;
        background: linear-gradient(135deg, #C7D06C, #E0E786);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        font-weight: 500;
        letter-spacing: 1.5px;
    }
    
    /* Glassmorphism cards - smaller sizing */
    .glass-card {
        background: rgba(199, 208, 108, 0.08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(224, 231, 134, 0.18);
        padding: 1.3rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(199, 208, 108, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(199, 208, 108, 0.25);
        border-color: rgba(224, 231, 134, 0.4);
    }
    
    /* Exercise cards with glassmorphism - smaller */
    .exercise-card {
        background: rgba(199, 208, 108, 0.06);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 14px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(224, 231, 134, 0.15);
        transition: all 0.3s ease;
    }
    
    .exercise-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 45px rgba(199, 208, 108, 0.3);
        border-color: rgba(224, 231, 134, 0.3);
    }
    
    /* Success/Warning/Info boxes - smaller */
    .success-box {
        background: linear-gradient(135deg, rgba(199, 208, 108, 0.15) 0%, rgba(224, 231, 134, 0.1) 100%);
        backdrop-filter: blur(15px);
        border-left: 4px solid #C7D06C;
        border-radius: 12px;
        padding: 1.3rem;
        margin: 1.2rem 0;
        box-shadow: 0 8px 25px rgba(199, 208, 108, 0.2);
        color: #E0E786;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.12) 0%, rgba(255, 152, 0, 0.08) 100%);
        backdrop-filter: blur(15px);
        border-left: 4px solid #ff6b6b;
        border-radius: 12px;
        padding: 1.3rem;
        margin: 1.2rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.2);
        color: #ff9a9a;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(199, 208, 108, 0.1) 0%, rgba(224, 231, 134, 0.05) 100%);
        backdrop-filter: blur(15px);
        border-left: 4px solid #E0E786;
        border-radius: 12px;
        padding: 1.3rem;
        margin: 1.2rem 0;
        box-shadow: 0 8px 25px rgba(224, 231, 134, 0.15);
        color: #E0E786;
    }
    
    /* Buttons - smaller */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 2.8rem;
        font-weight: 700;
        font-size: 0.85rem;
        border: 2px solid rgba(224, 231, 134, 0.3);
        background: linear-gradient(135deg, rgba(199, 208, 108, 0.15) 0%, rgba(224, 231, 134, 0.2) 100%);
        backdrop-filter: blur(10px);
        color: #E0E786;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(199, 208, 108, 0.2);
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(199, 208, 108, 0.5);
        background: linear-gradient(135deg, #C7D06C 0%, #E0E786 100%);
        color: #0a0a0a;
        border-color: #E0E786;
    }
    
    /* Metric cards - smaller */
    .metric-card {
        background: linear-gradient(135deg, rgba(199, 208, 108, 0.12) 0%, rgba(224, 231, 134, 0.08) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: #E0E786;
        padding: 1.3rem;
        border-radius: 14px;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 8px 32px rgba(199, 208, 108, 0.2);
        border: 1px solid rgba(224, 231, 134, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 50px rgba(199, 208, 108, 0.4);
        border-color: #E0E786;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.75rem;
        font-weight: 600;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #C7D06C;
    }
    
    .metric-card h1 {
        margin: 0.8rem 0 0 0;
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #C7D06C, #E0E786);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Video container */
    .video-container {
        width: 100%;
        max-width: 450px;
        margin: 1rem auto;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 15px 50px rgba(199, 208, 108, 0.3);
        border: 2px solid rgba(224, 231, 134, 0.3);
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .video-container video {
        width: 100% !important;
        height: auto !important;
        display: block;
    }
    
    /* Chat messages - smaller */
    .chat-user {
        background: linear-gradient(135deg, rgba(199, 208, 108, 0.15) 0%, rgba(224, 231, 134, 0.1) 100%);
        backdrop-filter: blur(15px);
        padding: 1.1rem;
        border-radius: 14px 14px 4px 14px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(199, 208, 108, 0.2);
        font-size: 0.9rem;
        border: 1px solid rgba(224, 231, 134, 0.2);
        color: #E0E786;
    }
    
    .chat-assistant {
        background: linear-gradient(135deg, rgba(149, 163, 179, 0.08) 0%, rgba(30, 30, 46, 0.6) 100%);
        backdrop-filter: blur(15px);
        padding: 1.1rem;
        border-radius: 14px 14px 14px 4px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(149, 163, 179, 0.2);
        font-size: 0.9rem;
        border: 1px solid rgba(149, 163, 179, 0.2);
        color: #b8c5d6;
    }
    
    /* Tabs - smaller */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(26, 26, 46, 0.4);
        backdrop-filter: blur(10px);
        padding: 0.8rem;
        border-radius: 12px;
        border: 1px solid rgba(224, 231, 134, 0.15);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 2.8rem;
        padding: 0 1.5rem;
        font-size: 0.8rem;
        font-weight: 700;
        border-radius: 10px;
        color: #C7D06C;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(199, 208, 108, 0.25) 0%, rgba(224, 231, 134, 0.15) 100%);
        backdrop-filter: blur(15px);
        color: #E0E786;
        box-shadow: 0 4px 20px rgba(199, 208, 108, 0.3);
        border: 1px solid rgba(224, 231, 134, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #C7D06C 0%, #E0E786 100%);
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(199, 208, 108, 0.4);
    }
    
    /* Input fields - smaller */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid rgba(224, 231, 134, 0.2);
        padding: 0.8rem;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        background: rgba(26, 26, 46, 0.4);
        backdrop-filter: blur(10px);
        color: #E0E786;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #E0E786;
        box-shadow: 0 0 0 3px rgba(224, 231, 134, 0.15);
        background: rgba(26, 26, 46, 0.6);
    }
    
    .streamlit-expanderHeader {
        background: rgba(199, 208, 108, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        font-weight: 700;
        color: #E0E786;
        font-size: 0.9rem;
        border: 1px solid rgba(224, 231, 134, 0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 50%, #0f3460 100%);
        color: #E0E786;
        border-right: 2px solid rgba(224, 231, 134, 0.2);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #E0E786;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #E0E786;
    }
    
    /* Badges - smaller */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 700;
        margin: 0.2rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        backdrop-filter: blur(10px);
    }
    
    .badge-primary {
        background: linear-gradient(135deg, rgba(199, 208, 108, 0.25) 0%, rgba(224, 231, 134, 0.15) 100%);
        color: #E0E786;
        border: 1px solid rgba(224, 231, 134, 0.3);
    }
    
    .badge-success {
        background: linear-gradient(135deg, rgba(168, 184, 72, 0.25) 0%, rgba(138, 154, 54, 0.15) 100%);
        color: #C7D06C;
        border: 1px solid rgba(199, 208, 108, 0.3);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.25) 0%, rgba(255, 152, 0, 0.15) 100%);
        color: #ff9a9a;
        border: 1px solid rgba(255, 107, 107, 0.3);
    }
    
    /* Typography - smaller sizes */
    h1, h2, h3 {
        color: #E0E786;
    }
    
    h1 {
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    h2 {
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    h3 {
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    p, li, span {
        font-size: 0.9rem;
        color: #b8c5d6;
        line-height: 1.6;
    }
    
    label {
        color: #E0E786 !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Icon styling */
    .icon-primary {
        color: #C7D06C;
        font-size: 1rem;
        margin-right: 0.4rem;
        vertical-align: middle;
    }
    
    .icon-accent {
        color: #ff6b6b;
        font-size: 1rem;
        margin-right: 0.4rem;
        vertical-align: middle;
    }
    
    /* Result display styling - smaller */
    .result-header {
        font-size: 2rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #C7D06C 0%, #E0E786 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1.5rem 0;
    }
    
    .result-status {
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1.2rem 0;
        backdrop-filter: blur(15px);
    }
    
    .result-perfect {
        background: linear-gradient(135deg, rgba(199, 208, 108, 0.2) 0%, rgba(224, 231, 134, 0.15) 100%);
        border: 2px solid rgba(199, 208, 108, 0.4);
        color: #E0E786;
        box-shadow: 0 10px 40px rgba(199, 208, 108, 0.3);
    }
    
    .result-mismatch {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 152, 0, 0.1) 100%);
        border: 2px solid rgba(255, 107, 107, 0.4);
        color: #ff9a9a;
        box-shadow: 0 10px 40px rgba(255, 107, 107, 0.3);
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
if 'processing' not in st.session_state:
    st.session_state.processing = False

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
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PoseClassifier(num_classes=5)
        model.load_state_dict(torch.load("best.pth", map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

@st.cache_resource
def setup_elasticsearch():
    try:
        es = Elasticsearch(
            cloud_id=os.environ["ES_CLOUD_ID"],
            api_key=os.environ["ES_API_KEY"]
        )
        return es
    except Exception as e:
        st.error(f"Elasticsearch connection error: {str(e)}")
        return None

@st.cache_resource
def setup_vertex_ai():
    try:
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
    except Exception as e:
        st.error(f"Vertex AI setup error: {str(e)}")
        return None

# Load resources with error handling
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
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_pose.Pose(static_image_mode=False) as pose_detector:
            results = pose_detector.process(frame_rgb)
            if not results.pose_landmarks:
                return None, None
            lm = results.pose_landmarks.landmark
            return [[l.x, l.y, l.z] for l in lm], results
    except Exception as e:
        return None, None

def detect_class(lm):
    try:
        if model is None or device is None:
            return "Unknown", 0.0
        lm_flat = [coord for point in lm for coord in point]
        input_tensor = torch.tensor(lm_flat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted].item()
            return class_names[predicted], confidence
    except Exception as e:
        return "Unknown", 0.0

def analyze_video(video_path, target_pose):
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Failed to open video file")
            return None
        
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
            progress = min(frame_count / max(total_frames, 1), 1.0)
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
                
                status = "CORRECT" if is_correct else "INCORRECT"
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
    except Exception as e:
        st.error(f"Video analysis error: {str(e)}")
        return None

def search_exercises(query):
    try:
        if es is None:
            return []
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
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def get_all_exercises():
    try:
        if es is None:
            return []
        result = es.search(
            index="pcos_exercises",
            body={"query": {"match_all": {}}, "size": 10}
        )
        return [hit['_source'] for hit in result['hits']['hits']]
    except Exception as e:
        return []

def chat_with_ai(user_message):
    try:
        if gemini_model is None:
            return "AI service is currently unavailable. Please try again later."
        
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
    except Exception as e:
        return f"I apologize, but I encountered an error. Please try again. Error: {str(e)}"

def get_exercise_image_path(exercise_name, exercise_id):
    try:
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
    except:
        return "logo.png"

# ==========================================
# MAIN APP LAYOUT
# ==========================================
col_logo, col_title = st.columns([0.5, 9.5])
with col_logo:
    st.image("logo.png", width=100)
with col_title:
    st.markdown('<h1 class="main-header">FLEXIFIT AI</h1>', unsafe_allow_html=True)

st.markdown('<p class="sub-header">Your AI-Powered PCOS/PCOD Exercise Coach with Real-Time Pose Detection</p>', unsafe_allow_html=True)

# ==========================================
# STATS ROW
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Exercises</h3>
        <h1>{len(st.session_state.exercise_history)}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_accuracy = np.mean([h['accuracy'] for h in st.session_state.exercise_history]) if st.session_state.exercise_history else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>Accuracy</h3>
        <h1>{avg_accuracy:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Messages</h3>
        <h1>{len(st.session_state.chat_history)}</h1>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>AI Power</h3>
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
    st.markdown('<h2 class="result-header">PCOS/PCOD Exercise Library</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 0.95rem; color: #b8c5d6; margin-bottom: 1.5rem;">Browse our curated collection of exercises specifically designed for PCOS/PCOD management</p>', unsafe_allow_html=True)
    
    search_query = st.text_input("Search exercises...", placeholder="Try: balance, beginner, stress relief, hormonal balance...")
    
    if search_query:
        exercises = search_exercises(search_query)
        
        if exercises:
            st.markdown(f'<h3 style="text-align: center; margin: 1.5rem 0;">Found {len(exercises)} exercises</h3>', unsafe_allow_html=True)
            
            cols = st.columns(2)
            
            for idx, ex in enumerate(exercises):
                with cols[idx % 2]:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    
                    image_name = get_exercise_image_path(ex.get('name', ''), ex.get('exercise_id', ''))
                    
                    if os.path.exists(image_name):
                        img = Image.open(image_name)
                        img = img.resize((300, 300))
                        st.image(img, width=300)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, rgba(199, 208, 108, 0.15) 0%, rgba(224, 231, 134, 0.08) 100%); 
                                    height: 180px; display: flex; align-items: center; justify-content: center;
                                    border-radius: 10px; color: #E0E786; font-size: 1rem; border: 2px solid rgba(224, 231, 134, 0.2);
                                    backdrop-filter: blur(10px);">
                            Exercise Image
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"### {ex['name']}")
                    
                    st.markdown(f"""
                    <span class="badge badge-primary">{ex['category']}</span>
                    <span class="badge badge-warning">{ex['difficulty']}</span>
                    <span class="badge badge-success">{ex['duration_seconds']}s</span>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Reps:** {ex['reps']}")
                    
                    with st.expander("View Details", expanded=False):
                        st.markdown(f"**Description:**\n{ex['description']}")
                        st.markdown("**PCOS/PCOD Benefits:**")
                        for benefit in ex['pcos_benefits']:
                            st.markdown(f"- {benefit}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 1.5rem;">
                <h3>No exercises found</h3>
                <p style="font-size: 0.95rem;">Try different keywords like "balance", "beginner", "stress relief", or "hormonal balance"</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 2rem;">
            <h2>Search for PCOS/PCOD Exercises</h2>
            <p style="font-size: 0.95rem;">Type keywords like "balance", "beginner", "stress relief", or "hormonal balance" to find exercises</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: ANALYZE VIDEO
# ==========================================
with tab2:
    st.markdown('<h2 class="result-header">Upload & Analyze Your Exercise Video</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3 style="margin-top: 0;">How it works:</h3>
    <ol style="font-size: 0.9rem; line-height: 1.8;">
        <li><b>Choose</b> the exercise you're performing from the dropdown</li>
        <li><b>Upload</b> your video (MP4, MOV, AVI format)</li>
        <li><b>Analyze</b> - Our AI will detect your pose in real-time</li>
        <li><b>Download</b> the annotated video with visual feedback</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3>Select Target Exercise</h3>', unsafe_allow_html=True)
        
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
        <h3 style="margin: 0;">Target Exercise Selected</h3>
        <h2 style="margin: 0.8rem 0 0 0; color: #E0E786; font-size: 1.5rem;">{selected_display}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3>Upload Your Video</h3>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi'],
            help="Upload a video showing your full body performing the exercise"
        )
    
    if uploaded_video is not None:
        st.markdown("---")
        
        col_preview, col_analyze = st.columns([1, 1])
        
        with col_preview:
            st.markdown('<h3>Your Uploaded Video</h3>', unsafe_allow_html=True)
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.video(uploaded_video)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_analyze:
            st.markdown('<h3>Ready to Analyze</h3>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
            <h3 style="margin-top: 0;">Analysis Details</h3>
            <p style="font-size: 0.9rem;"><b>Target Exercise:</b> {selected_display}</p>
            <p style="font-size: 0.9rem;"><b>AI Model:</b> Custom Pose Classifier</p>
            <p style="font-size: 0.9rem;"><b>Accuracy:</b> 92% on validation set</p>
            <p style="font-size: 0.9rem;"><b>Processing:</b> Real-time frame analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start AI Analysis", type="primary", use_container_width=True):
                if not st.session_state.processing:
                    st.session_state.processing = True
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    tfile.close()
                    
                    with st.spinner("AI is analyzing your video... Please wait"):
                        results = analyze_video(tfile.name, target_pose)
                    
                    try:
                        os.unlink(tfile.name)
                    except:
                        pass
                    
                    st.session_state.processing = False
                    
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
                        st.markdown('<h2 class="result-header">Analysis Results</h2>', unsafe_allow_html=True)
                        
                        if results['match']:
                            st.markdown("""
                            <div class="result-status result-perfect">
                            <h2 style="margin: 0; font-size: 1.8rem;">PERFECT MATCH</h2>
                            <p style="margin: 1rem 0 0 0; font-size: 1rem; line-height: 1.6;">
                            Excellent work! Your pose matches the target exercise perfectly. Keep up the great form!
                            </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-status result-mismatch">
                            <h2 style="margin: 0; font-size: 1.8rem;">Different Pose Detected</h2>
                            <p style="margin: 1rem 0 0 0; font-size: 0.95rem; line-height: 1.6;">
                            <b>Target Exercise:</b> {selected_display}<br>
                            <b>Detected Exercise:</b> {exercise_mapping.get(results['detected_pose'], results['detected_pose'])}<br><br>
                            Don't worry! Check the annotated video below to see where adjustments are needed.
                            </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.markdown(f"""
                            <div class="metric-card">
                            <h3>Accuracy</h3>
                            <h1>{results['accuracy']:.1f}%</h1>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col2:
                            st.markdown(f"""
                            <div class="metric-card">
                            <h3>Confidence</h3>
                            <h1>{results['confidence']*100:.1f}%</h1>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col3:
                            st.markdown(f"""
                            <div class="metric-card">
                            <h3>Frames</h3>
                            <h1>{results['total_frames']}</h1>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown('<h3 style="text-align: center;">Annotated Video with AI Feedback</h3>', unsafe_allow_html=True)
                        st.markdown('<p style="text-align: center; font-size: 0.95rem; margin-bottom: 1.2rem;"><span style="color: #4CAF50; font-weight: 700;">Green</span> = Correct Pose | <span style="color: #f44336; font-weight: 700;">Red</span> = Incorrect Pose</p>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                        st.video(results['output_path'])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        col_action1, col_action2 = st.columns(2)
                        
                        with col_action1:
                            with open(results['output_path'], 'rb') as video_file:
                                video_bytes = video_file.read()
                                st.download_button(
                                    label="Download Annotated Video",
                                    data=video_bytes,
                                    file_name=f"flexifit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                        
                        with col_action2:
                            if st.button("Analyze Another Video", use_container_width=True):
                                st.session_state.analyzed_video_path = None
                                st.rerun()

# ==========================================
# TAB 3: AI CHAT
# ==========================================
with tab3:
    st.markdown('<h2 class="result-header">Chat with Your AI Exercise Coach</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3 style="margin-top: 0;">Ask me anything about PCOS/PCOD exercises</h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
        <div>
            <b style="font-size: 0.95rem;">Exercise Questions:</b>
            <ul style="margin: 0.8rem 0; font-size: 0.85rem;">
                <li>"What exercises help with PCOS?"</li>
                <li>"How to improve my plank form?"</li>
                <li>"Best poses for stress relief?"</li>
            </ul>
        </div>
        <div>
            <b style="font-size: 0.95rem;">Health & Wellness:</b>
            <ul style="margin: 0.8rem 0; font-size: 0.85rem;">
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
            <div style="text-align: center; padding: 2.5rem; color: #b8c5d6;">
                <h2 style="color: #E0E786; margin-top: 1rem;">Welcome to AI Coach Chat</h2>
                <p style="font-size: 1rem; margin-top: 0.8rem;">Start a conversation by typing your question below.</p>
            </div>
            """, unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-user">
                <b style="color: #E0E786; font-size: 0.95rem;">You:</b><br>
                <p style="margin: 0.6rem 0 0 0; font-size: 0.9rem; color: #E0E786; line-height: 1.5;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-assistant">
                <b style="color: #b8c5d6; font-size: 0.95rem;">AI Coach:</b><br>
                <p style="margin: 0.6rem 0 0 0; font-size: 0.9rem; color: #b8c5d6; line-height: 1.5;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    user_input = st.text_input("Type your message...", key="chat_input", placeholder="Ask me anything about PCOS exercises, nutrition, or wellness...")
    
    col_send, col_clear = st.columns([3, 1])
    
    with col_send:
        if st.button("Send Message", use_container_width=True, type="primary"):
            if user_input:
                if not st.session_state.processing:
                    st.session_state.processing = True
                    
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
                    
                    st.session_state.processing = False
                    st.rerun()
            else:
                st.warning("Please type a message first")
    
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ==========================================
# TAB 4: PROGRESS HISTORY
# ==========================================
with tab4:
    st.markdown('<h2 class="result-header">Your Progress & History</h2>', unsafe_allow_html=True)
    
    history_tab1, history_tab2 = st.tabs(["Exercise Analytics", "Chat History"])
    
    with history_tab1:
        if st.session_state.exercise_history:
            st.markdown(f"<h3 style='text-align: center; margin: 1.5rem 0;'>Total Workouts Completed: {len(st.session_state.exercise_history)}</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_acc = np.mean([h['accuracy'] for h in st.session_state.exercise_history])
                st.markdown(f"""
                <div class="metric-card">
                <h3>Avg Accuracy</h3>
                <h1>{avg_acc:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_conf = np.mean([h['confidence'] for h in st.session_state.exercise_history])
                st.markdown(f"""
                <div class="metric-card">
                <h3>Avg Confidence</h3>
                <h1>{avg_conf*100:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                matches = sum(1 for h in st.session_state.exercise_history if h['target_pose'] == h['detected_pose'])
                success_rate = (matches / len(st.session_state.exercise_history)) * 100
                st.markdown(f"""
                <div class="metric-card">
                <h3>Success Rate</h3>
                <h1>{success_rate:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<h3 style="text-align: center;">Workout History</h3>', unsafe_allow_html=True)
            
            for idx, record in enumerate(reversed(st.session_state.exercise_history)):
                match_status = record['target_pose'] == record['detected_pose']
                
                status_text = "Perfect Match" if match_status else "Different Pose"
                
                with st.expander(f"{record['timestamp']} - {record['target_pose']} - {status_text}", expanded=(idx==0)):
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Target Exercise**")
                        st.markdown(f"<h3 style='color: #E0E786;'>{record['target_pose']}</h3>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Detected Exercise**")
                        st.markdown(f"<h3 style='color: #E0E786;'>{record['detected_pose']}</h3>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("**Result**")
                        if match_status:
                            st.markdown(f'<h3 style="color: #C7D06C;">Perfect Match</h3>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<h3 style="color: #ff6b6b;">Different Pose</h3>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    progress_col1, progress_col2 = st.columns(2)
                    
                    with progress_col1:
                        st.markdown("**Accuracy Score**")
                        st.progress(record['accuracy'] / 100)
                        st.caption(f"{record['accuracy']:.1f}%")
                    
                    with progress_col2:
                        st.markdown("**AI Confidence**")
                        st.progress(record['confidence'])
                        st.caption(f"{record['confidence']*100:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 2.5rem;">
                <h2 style="color: #E0E786; margin-top: 1rem;">No Exercise History Yet</h2>
                <p style="font-size: 1rem; margin: 1.2rem 0;">
                Upload and analyze a video in the "Analyze Video" tab to start tracking your progress
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with history_tab2:
        if st.session_state.chat_history:
            st.markdown(f"<h3 style='text-align: center; margin: 1.5rem 0;'>Total Conversations: {len(st.session_state.chat_history) // 2}</h3>", unsafe_allow_html=True)
            
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-user">
                    <b style="color: #E0E786; font-size: 0.95rem;">You:</b><br>
                    <p style="margin: 0.6rem 0 0 0; font-size: 0.9rem; line-height: 1.5;">{message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-assistant">
                    <b style="color: #b8c5d6; font-size: 0.95rem;">AI Coach:</b><br>
                    <p style="margin: 0.6rem 0 0 0; font-size: 0.9rem; line-height: 1.5;">{message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 2.5rem;">
                <h2 style="color: #E0E786; margin-top: 1rem;">No Chat History Yet</h2>
                <p style="font-size: 1rem; margin: 1.2rem 0;">
                Start a conversation with the AI Coach in the "AI Coach Chat" tab
                </p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.2rem 0; margin-bottom: 1.5rem;">
    """, unsafe_allow_html=True)
    
    st.image("logo.png", width=80)
    
    st.markdown("""
        <h2 style="color: #E0E786; margin: 0.8rem 0; font-weight: 900; font-size: 1.5rem;">FLEXIFIT AI</h2>
        <p style="color: #b8c5d6; margin: 0; font-size: 0.85rem;">PCOS/PCOD Exercise Coach</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="color: #E0E786;">
        <h3 style="color: #E0E786; margin-bottom: 1.2rem;">Powered By</h3>
        <ul style="list-style: none; padding: 0;">
            <li style="padding: 0.6rem 0;">
                <b style="font-size: 0.9rem;">MediaPipe</b><br>
                <span style="opacity: 0.8; color: #b8c5d6; font-size: 0.8rem;">Real-time pose detection</span>
            </li>
            <li style="padding: 0.6rem 0;">
                <b style="font-size: 0.9rem;">Custom ML Model</b><br>
                <span style="opacity: 0.8; color: #b8c5d6; font-size: 0.8rem;">92% accuracy classification</span>
            </li>
            <li style="padding: 0.6rem 0;">
                <b style="font-size: 0.9rem;">Elasticsearch</b><br>
                <span style="opacity: 0.8; color: #b8c5d6; font-size: 0.8rem;">Smart exercise search</span>
            </li>
            <li style="padding: 0.6rem 0;">
                <b style="font-size: 0.9rem;">Vertex AI Gemini</b><br>
                <span style="opacity: 0.8; color: #b8c5d6; font-size: 0.8rem;">Intelligent coaching</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("Clear All History", use_container_width=True):
        st.session_state.exercise_history = []
        st.session_state.chat_history = []
        st.success("All history cleared")
        st.rerun()
