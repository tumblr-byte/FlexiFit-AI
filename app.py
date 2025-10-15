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
<style>
.icon-light {
    color: #b0b0b0;
    font-size: 22px;
    margin-right: 8px;
    vertical-align: middle;
}

.icon-primary {
    color: #4CAF50; /* greenish highlight */
    font-size: 22px;
    margin-right: 8px;
    vertical-align: middle;
}

.icon-accent {
    color: #ff6b6b; /* reddish accent */
    font-size: 22px;
    margin-right: 8px;
    vertical-align: middle;
}

.icon-hover:hover {
    transform: scale(1.15);
    transition: 0.3s;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background: linear-gradient(135deg, ##dee276 0%, #c3cfe2 100%);
        padding: 2rem 1rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1.5rem 0;
        animation: fadeInDown 1s ease-out;
    }
    
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: ##dee276;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeInUp 1s ease-out;
    }
    
    .exercise-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .exercise-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(45, 106, 79, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .exercise-card:hover::before {
        left: 100%;
    }
    
    .exercise-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(45, 106, 79, 0.2);
        border-color: #52b788;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: none;
        border-left: 5px solid #2d6a4f;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        font-size: 1rem;
        box-shadow: 0 3px 12px rgba(45, 106, 79, 0.15);
        animation: slideInLeft 0.5s ease-out;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: none;
        border-left: 5px solid #ffc107;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 3px 12px rgba(255, 193, 7, 0.15);
        animation: slideInLeft 0.5s ease-out;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: none;
        border-left: 5px solid #52b788;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 3px 12px rgba(82, 183, 136, 0.15);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 3px 12px rgba(45, 106, 79, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(45, 106, 79, 0.5);
        background: linear-gradient(135deg, #52b788 0%, #2d6a4f 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(45, 106, 79, 0.3);
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 25px rgba(45, 106, 79, 0.5);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        font-weight: 400;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .metric-card h1 {
        margin: 0.8rem 0 0 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .video-container {
        width: 100%;
        max-width: 400px;
        margin: 1rem auto;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        border: 2px solid #52b788;
        background: #f8f9fa;
    }
    
    .video-container video {
        width: 100% !important;
        height: auto !important;
        display: block;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.2rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.8rem 0;
        box-shadow: 0 3px 10px rgba(45, 106, 79, 0.15);
        animation: slideInRight 0.3s ease-out;
    }
    
    .chat-assistant {
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        padding: 1.2rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.8rem 0;
        box-shadow: 0 3px 10px rgba(82, 183, 136, 0.15);
        animation: slideInLeft 0.3s ease-out;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: rgba(255, 255, 255, 0.5);
        padding: 0.8rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        color: #2d6a4f;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
        color: white;
        box-shadow: 0 3px 12px rgba(45, 106, 79, 0.3);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
        border-radius: 10px;
    }
    
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #52b788;
        box-shadow: 0 0 0 3px rgba(82, 183, 136, 0.1);
    }
    
    .streamlit-expanderHeader {
        background: rgba(82, 183, 136, 0.1);
        border-radius: 10px;
        font-weight: 600;
        color: #2d6a4f;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
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
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d6a4f 0%, #52b788 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white;
    }
    
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    
    .badge-primary {
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
        color: white;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        color: white;
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
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.image("logo.png", width=100)
with col_title:
    st.markdown('<h1 class="main-header">FlexiFit AI</h1>', unsafe_allow_html=True)

st.markdown('<p class="sub-header">Your AI-Powered PCOS/PCOD Exercise Coach with Real-Time Pose Detection</p>', unsafe_allow_html=True)

# ==========================================
# STATS ROW
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Exercises</h3>
        <h1>{len(st.session_state.exercise_history)}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_accuracy = np.mean([h['accuracy'] for h in st.session_state.exercise_history]) if st.session_state.exercise_history else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>Avg Accuracy</h3>
        <h1>{avg_accuracy:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Chat Messages</h3>
        <h1>{len(st.session_state.chat_history)}</h1>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>AI Accuracy</h3>
        <h1>92%</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
 st.markdown('<i class="fa-solid fa-dumbbell"></i> **Workout Tracker**', unsafe_allow_html=True)
st.markdown('<i class="fa-solid fa-video icon-primary"></i> **Analyze Video**', unsafe_allow_html=True)
st.markdown('<i class="fa-solid fa-comments icon-accent"></i> **AI Coach Chat**', unsafe_allow_html=True)
st.markdown('<i class="fa-solid fa-chart-line icon-light icon-hover"></i> **Progress Stats**', unsafe_allow_html=True)

])

# ==========================================
# TAB 1: EXERCISE LIBRARY
# ==========================================
with tab1:
    st.markdown("""
<h2><i class="fa-solid fa-book icon-title"></i> PCOS/PCOD Exercise Library</h2>
""", unsafe_allow_html=True)
    st.markdown("Browse our curated collection of exercises specifically designed for PCOS/PCOD management")
    
    search_query = st.text_input("Search exercises...", placeholder="Try: balance, beginner, stress relief, hormonal balance...")
    
    if search_query:
        exercises = search_exercises(search_query)
        
        if exercises:
           st.markdown(f"""
<h3><i class="fa-solid fa-bullseye icon"></i> Found {len(exercises)} exercises</h3>
""", unsafe_allow_html=True)
            
            cols = st.columns(2)
            
            for idx, ex in enumerate(exercises):
                with cols[idx % 2]:
                    st.markdown('<div class="exercise-card">', unsafe_allow_html=True)
                    
                    image_name = get_exercise_image_path(ex.get('name', ''), ex.get('exercise_id', ''))
                    
                    if os.path.exists(image_name):
                        img = Image.open(image_name)
                        img = img.resize((350, 350))
                        st.image(img, width=350)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%); 
                                    height: 180px; display: flex; align-items: center; justify-content: center;
                                    border-radius: 10px; color: white; font-size: 1.3rem;">
                            
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"### {ex['name']}")
                    
                    st.markdown(f"""
                    <span class="badge badge-primary">{ex['category']}</span>
                    <span class="badge badge-warning">{ex['difficulty']}</span>
                    <span class="badge badge-success">{ex['duration_seconds']}s</span>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Reps:** {ex['reps']}")
                    
                   with st.expander('<i class="fa-solid fa-book-open icon"></i> View Details', expanded=False):
                    st.markdown(f"""
                         <p><i class="fa-solid fa-align-left icon"></i> <b>Description:</b><br>{ex['description']}</p>
                                   """, unsafe_allow_html=True)
    
                    st.markdown("""
                          <p><i class="fa-solid fa-heart-pulse icon-benefit"></i> <b>PCOS/PCOD Benefits:</b></p>
                                   """, unsafe_allow_html=True)
                        for benefit in ex['pcos_benefits']:
                            st.markdown(f"• {benefit}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 2rem;">
                <h3> No exercises found</h3>
                <p style="font-size: 1rem; margin: 1rem 0;">
                Try different keywords like "balance", "beginner", "stress relief", or "hormonal balance"
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 2rem;">
            <h3> Search for PCOS/PCOD Exercises</h3>
            <p style="font-size: 1rem; margin: 1rem 0;">
            Type keywords like "balance", "beginner", "stress relief", or "hormonal balance" to find exercises!
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: ANALYZE VIDEO
# ==========================================
with tab2:
    st.markdown("## 🎬 Upload & Analyze Your Exercise Video")
    
    st.markdown("""
    <div class="info-box">
    <h3 style="margin-top: 0;">📹 How it works:</h3>
    <ol style="font-size: 1.1rem; line-height: 2;">
        <li><b>Choose</b> the exercise you're performing from the dropdown</li>
        <li><b>Upload</b> your video (MP4, MOV, AVI format)</li>
        <li><b>Analyze</b> - Our AI will detect your pose in real-time</li>
        <li><b>Download</b> the annotated video with visual feedback!</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1️⃣ Select Target Exercise")
        
        exercise_mapping = {
            "Downdog": "Downward Facing Dog 🐕",
            "Plank": "Plank Pose 💪",
            "Warrior2": "Warrior 2 ⚔️",
            "Modified_Tree": "Modified Tree Pose 🌳",
            "Standard_Tree": "Standard Tree Pose 🌲"
        }
        
        display_names = list(exercise_mapping.values())
        selected_display = st.selectbox("Choose your exercise:", display_names, key="exercise_select")
        
        reverse_mapping = {v: k for k, v in exercise_mapping.items()}
        target_pose = reverse_mapping[selected_display]
        
        st.markdown(f"""
        <div class="success-box">
        <h3 style="margin: 0;">✅ Target Exercise Selected</h3>
        <h2 style="margin: 0.5rem 0 0 0; color: #28a745;">{selected_display}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 2️⃣ Upload Your Video")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi'],
            help="Upload a video showing your full body performing the exercise"
        )
    
    if uploaded_video is not None:
        st.markdown("---")
        
        col_preview, col_analyze = st.columns([1, 1])
        
        with col_preview:
            st.markdown("### 📹 Your Uploaded Video")
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.video(uploaded_video)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_analyze:
            st.markdown("### 🔍 Ready to Analyze!")
            st.markdown(f"""
            <div class="info-box">
            <h3 style="margin-top: 0;">Analysis Details</h3>
            <p><b>Target Exercise:</b> {selected_display}</p>
            <p><b>AI Model:</b> Custom Pose Classifier</p>
            <p><b>Accuracy:</b> 92% on validation set</p>
            <p><b>Processing:</b> Real-time frame analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                
                with st.spinner("🤖 AI is analyzing your video... This may take a moment!"):
                    results = analyze_video(tfile.name, target_pose)
                
                # Clean up temporary file
                os.unlink(tfile.name)
                
                if results:
                    st.session_state.analyzed_video_path = results['output_path']
                    
                    # Add to history
                    st.session_state.exercise_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'target_pose': selected_display,
                        'detected_pose': exercise_mapping.get(results['detected_pose'], results['detected_pose']),
                        'accuracy': results['accuracy'],
                        'confidence': results['confidence']
                    })
                    
                    # Keep only last 50 entries
                    if len(st.session_state.exercise_history) > 50:
                        st.session_state.exercise_history = st.session_state.exercise_history[-50:]
                    
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    if results['match']:
                        st.markdown("""
                        <div class="success-box">
                        <h2 style="color: #28a745; margin: 0;">🎉 PERFECT MATCH!</h2>
                        <p style="margin: 1rem 0 0 0; font-size: 1.2rem;">
                        Excellent work! Your pose matches the target exercise perfectly. Keep up the great form!
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                        <h2 style="color: #856404; margin: 0;">⚠️ Different Pose Detected</h2>
                        <p style="margin: 1rem 0 0 0; font-size: 1.1rem;">
                        <b>Target Exercise:</b> {selected_display}<br>
                        <b>Detected Exercise:</b> {exercise_mapping.get(results['detected_pose'], results['detected_pose'])}<br><br>
                        Don't worry! Check the annotated video below to see where adjustments are needed.
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3>Accuracy Score</h3>
                        <h1>{results['accuracy']:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3>AI Confidence</h3>
                        <h1>{results['confidence']*100:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3>Frames Analyzed</h3>
                        <h1>{results['total_frames']}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### 🎥 Annotated Video with AI Feedback")
                    st.markdown("**Green** = Correct Pose | **Red** = Incorrect Pose")
                    
                    st.markdown('<div class="video-container">', unsafe_allow_html=True)
                    st.video(results['output_path'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    col_action1, col_action2 = st.columns(2)
                    
                    with col_action1:
                        with open(results['output_path'], 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.download_button(
                                label="📥 Download Annotated Video",
                                data=video_bytes,
                                file_name=f"flexifit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                    
                    with col_action2:
                        if st.button("🔄 Analyze Another Video", use_container_width=True):
                            st.session_state.analyzed_video_path = None
                            st.rerun()

# ==========================================
# TAB 3: AI CHAT
# ==========================================
with tab3:
    st.markdown("## 💬 Chat with Your AI Exercise Coach")
    
    st.markdown("""
    <div class="info-box">
    <h3 style="margin-top: 0;">💡 Ask me anything about PCOS/PCOD exercises!</h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
        <div>
            <b>🏋️ Exercise Questions:</b>
            <ul style="margin: 0.5rem 0;">
                <li>"What exercises help with PCOS?"</li>
                <li>"How to improve my plank form?"</li>
                <li>"Best poses for stress relief?"</li>
            </ul>
        </div>
        <div>
            <b>🌟 Health & Wellness:</b>
            <ul style="margin: 0.5rem 0;">
                <li>"Benefits of Tree Pose?"</li>
                <li>"How often should I exercise?"</li>
                <li>"Tips for better hormonal balance?"</li>
            </ul>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #999;">
                <h2>👋 Welcome to AI Coach Chat!</h2>
                <p style="font-size: 1.2rem;">Start a conversation by typing your question below.</p>
            </div>
            """, unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-user">
                <b style="color: #2196f3; font-size: 1.1rem;">👤 You:</b><br>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-assistant">
                <b style="color: #9c27b0; font-size: 1.1rem;">🤖 AI Coach:</b><br>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chat input
    user_input = st.text_input("💭 Type your message...", key="chat_input", placeholder="Ask me anything about PCOS exercises, nutrition, or wellness...")
    
    col_send, col_clear = st.columns([3, 1])
    
    with col_send:
        if st.button("📤 Send Message", use_container_width=True, type="primary"):
            if user_input:
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Keep only last 100 messages
                if len(st.session_state.chat_history) > 100:
                    st.session_state.chat_history = st.session_state.chat_history[-100:]
                
                with st.spinner("🤖 AI Coach is thinking..."):
                    response = chat_with_ai(user_input)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
            else:
                st.warning("⚠️ Please type a message first!")
    
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ==========================================
# TAB 4: PROGRESS HISTORY
# ==========================================
with tab4:
    st.markdown("## 📊 Your Progress & History")
    
    history_tab1, history_tab2 = st.tabs(["🏋️ Exercise Analytics", "💬 Chat History"])
    
    with history_tab1:
        if st.session_state.exercise_history:
            st.markdown(f"### 📈 Total Workouts Completed: {len(st.session_state.exercise_history)}")
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_acc = np.mean([h['accuracy'] for h in st.session_state.exercise_history])
                st.markdown(f"""
                <div class="metric-card">
                <h3>Average Accuracy</h3>
                <h1>{avg_acc:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_conf = np.mean([h['confidence'] for h in st.session_state.exercise_history])
                st.markdown(f"""
                <div class="metric-card">
                <h3>Average Confidence</h3>
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
            st.markdown("### 📜 Workout History")
            
            for idx, record in enumerate(reversed(st.session_state.exercise_history)):
                match_status = record['target_pose'] == record['detected_pose']
                
                with st.expander(f"{'✅' if match_status else '⚠️'} {record['timestamp']} - {record['target_pose']}", expanded=(idx==0)):
                    st.markdown('<div class="exercise-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Target Exercise**")
                        st.markdown(f"<h3>{record['target_pose']}</h3>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Detected Exercise**")
                        st.markdown(f"<h3>{record['detected_pose']}</h3>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("**Result**")
                        if match_status:
                            st.markdown('<h3 style="color: #28a745;">✅ Perfect Match</h3>', unsafe_allow_html=True)
                        else:
                            st.markdown('<h3 style="color: #ffc107;">⚠️ Different Pose</h3>', unsafe_allow_html=True)
                    
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
            <div class="info-box" style="text-align: center; padding: 3rem;">
                <h2>📊 No Exercise History Yet</h2>
                <p style="font-size: 1.2rem; margin: 1rem 0;">
                Upload and analyze a video in the "Analyze Video" tab to start tracking your progress!
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with history_tab2:
        if st.session_state.chat_history:
            st.markdown(f"### 💬 Total Conversations: {len(st.session_state.chat_history) // 2}")
            
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-user">
                    <b style="color: #2196f3;">👤 You:</b> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-assistant">
                    <b style="color: #9c27b0;">🤖 AI Coach:</b> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 3rem;">
                <h2>💬 No Chat History Yet</h2>
                <p style="font-size: 1.2rem; margin: 1rem 0;">
                Start a conversation with the AI Coach in the "AI Coach Chat" tab!
                </p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">
    """, unsafe_allow_html=True)
    
    st.image("logo.png", width=80)
    
    st.markdown("""
        <h2 style="color: white; margin: 0.5rem 0;">FlexiFit AI</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 0;">PCOS/PCOD Exercise Coach</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="color: white;">
        <h3 style="color: white;">Powered By</h3>
        <ul style="list-style: none; padding: 0;">
            <li style="padding: 0.5rem 0;">
                <b>MediaPipe</b><br>
                <span style="opacity: 0.8;">Real-time pose detection</span>
            </li>
            <li style="padding: 0.5rem 0;">
                <b>Custom ML Model</b><br>
                <span style="opacity: 0.8;">92% accuracy classification</span>
            </li>
            <li style="padding: 0.5rem 0;">
                <b>Elasticsearch</b><br>
                <span style="opacity: 0.8;">Smart exercise search</span>
            </li>
            <li style="padding: 0.5rem 0;">
                <b>Vertex AI Gemini</b><br>
                <span style="opacity: 0.8;">Intelligent coaching</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All History", use_container_width=True):
        st.session_state.exercise_history = []
        st.session_state.chat_history = []
        st.success("✅ All history cleared!")
        st.rerun()


