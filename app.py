import streamlit as st
import cv2
import numpy as np
import torch

import torch.nn as nn
import mediapipe as mp
from elasticsearch import Elasticsearch
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
import tempfile
import os
from datetime import datetime
from PIL import Image
import json


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="FlexiFit AI - PCOS/PCOD Exercise Coach",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .exercise-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        transition: all 0.3s;
    }
    .exercise-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    .success-box {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.1rem;
    }
    .warning-box {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if 'exercise_history' not in st.session_state:
    st.session_state.exercise_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analyzed_video_path' not in st.session_state:
    st.session_state.analyzed_video_path = None

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_pose_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    model = PoseClassifier(num_classes=5)
    model.load_state_dict(torch.load("best.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model, device



@st.cache_resource
def setup_elasticsearch():
    return Elasticsearch(
        cloud_id=os.environ["ES_CLOUD_ID"],
        api_key=os.environ["ES_API_KEY"]
    )

@st.cache_resource
def setup_vertex_ai():
    service_account_info = json.loads(os.environ["VERTEX_SERVICE_ACCOUNT"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    aiplatform.init(
        project=os.environ["VERTEX_PROJECT_ID"],
        location=os.environ["VERTEX_LOCATION"],
        credentials=credentials
    )
    return GenerativeModel("gemini-2.5-flash")



model, device = load_pose_model()
es = setup_elasticsearch()
gemini_model = setup_vertex_ai()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

class_names = ["Downdog", "Plank", "Warrior2", "Modified_Tree", "Standard_Tree"]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        return None, None
    lm = results.pose_landmarks.landmark
    return [[l.x, l.y, l.z] for l in lm], results

def detect_class(lm):
    lm_flat = [coord for point in lm for coord in point]
    input_tensor = torch.tensor(lm_flat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted].item()
        return class_names[predicted], confidence

def analyze_video(video_path, target_pose):
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
        from collections import Counter
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
    result = es.search(
        index="pcos_exercises",
        body={"query": {"match_all": {}}, "size": 10}
    )
    return [hit['_source'] for hit in result['hits']['hits']]

def chat_with_ai(user_message):
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

# ==========================================
# MAIN APP
# ==========================================
st.markdown('<h1 class="main-header">🧘‍♀️ FlexiFit AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your AI-Powered PCOS/PCOD Exercise Coach</p>', unsafe_allow_html=True)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏋️ Exercise Library", 
    "🎬 Analyze Video", 
    "💬 AI Chat", 
    "📊 My History"
])

# ==========================================
# TAB 1: EXERCISE LIBRARY
# ==========================================
with tab1:
    st.header("📚 PCOS/PCOD Exercise Library")
    
    search_query = st.text_input("🔍 Search exercises...", placeholder="Try: balance, beginner, stress relief...")
    
    if search_query:
        exercises = search_exercises(search_query)
    else:
        exercises = get_all_exercises()
    
    st.markdown(f"### Found {len(exercises)} exercises")
    
    cols = st.columns(2)
    
    for idx, ex in enumerate(exercises):
        with cols[idx % 2]:
            with st.container():
                st.markdown('<div class="exercise-card">', unsafe_allow_html=True)
                
                # Try to load image
                image_name = ex['exercise_id'] + ".jpg"
                if os.path.exists(image_name):
                    img = Image.open(image_name)
                    st.image(img, use_column_width=True)
                
                st.markdown(f"### {ex['name']}")
                st.markdown(f"**Category:** `{ex['category']}` | **Difficulty:** `{ex['difficulty']}`")
                st.markdown(f"**Duration:** {ex['duration_seconds']}s | **Reps:** {ex['reps']}")
                
                with st.expander("📖 Details"):
                    st.markdown(f"**Description:**\n{ex['description']}")
                    st.markdown("**PCOS/PCOD Benefits:**")
                    for benefit in ex['pcos_benefits']:
                        st.markdown(f"- {benefit}")
                
                st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: ANALYZE VIDEO
# ==========================================
with tab2:
    st.header("🎬 Upload & Analyze Your Exercise Video")
    
    st.markdown("""
    <div class="info-box">
    📹 <b>How it works:</b><br>
    1. Choose the exercise you're performing<br>
    2. Upload your video (MP4, MOV, AVI)<br>
    3. Our AI analyzes your posture in real-time<br>
    4. Get instant feedback with visual overlays!
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1️⃣ Select Target Exercise")
        
        # Map class names to display names
        exercise_mapping = {
            "Downdog": "Downward Facing Dog",
            "Plank": "Plank Pose",
            "Warrior2": "Warrior 2",
            "Modified_Tree": "Modified Tree Pose",
            "Standard_Tree": "Standard Tree Pose"
        }
        
        display_names = list(exercise_mapping.values())
        selected_display = st.selectbox("Choose your exercise:", display_names)
        
        # Reverse mapping
        reverse_mapping = {v: k for k, v in exercise_mapping.items()}
        target_pose = reverse_mapping[selected_display]
        
        st.success(f"✅ Target Exercise: **{selected_display}**")
    
    with col2:
        st.subheader("2️⃣ Upload Video")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi'],
            help="Upload a video showing your full body performing the exercise"
        )
    
    if uploaded_video is not None:
        st.markdown("---")
        
        col_preview, col_analyze = st.columns([1, 1])
        
        with col_preview:
            st.subheader("📹 Your Uploaded Video")
            st.video(uploaded_video)
        
        with col_analyze:
            st.subheader("🔍 Ready to Analyze!")
            st.info(f"**Target Exercise:** {selected_display}")
            
            if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                
                with st.spinner("🤖 AI is analyzing your video... Please wait!"):
                    results = analyze_video(tfile.name, target_pose)
                
                if results:
                    st.session_state.analyzed_video_path = results['output_path']
                    
                    # Save to history
                    st.session_state.exercise_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'target_pose': selected_display,
                        'detected_pose': exercise_mapping.get(results['detected_pose'], results['detected_pose']),
                        'accuracy': results['accuracy'],
                        'confidence': results['confidence']
                    })
                    
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    if results['match']:
                        st.markdown("""
                        <div class="success-box">
                        <h2 style="color: #28a745; margin: 0;">✅ PERFECT MATCH!</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                        Great job! Your pose matches the target exercise.
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                        <h2 style="color: #856404; margin: 0;">⚠️ Different Pose Detected</h2>
                        <p style="margin: 0.5rem 0 0 0;">
                        <b>Target:</b> {selected_display}<br>
                        <b>Detected:</b> {exercise_mapping.get(results['detected_pose'], results['detected_pose'])}
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3 style="margin: 0;">Accuracy</h3>
                        <h1 style="margin: 0.5rem 0 0 0;">{results['accuracy']:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3 style="margin: 0;">Confidence</h3>
                        <h1 style="margin: 0.5rem 0 0 0;">{results['confidence']*100:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3 style="margin: 0;">Frames</h3>
                        <h1 style="margin: 0.5rem 0 0 0;">{results['total_frames']}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("🎥 Analyzed Video (with AI Overlay)")
                    st.video(results['output_path'])
                    
                    st.markdown("---")
                    
                    col_action1, col_action2 = st.columns(2)
                    
                    with col_action1:
                        if st.button("🔄 Analyze Another Video", use_container_width=True):
                            st.session_state.analyzed_video_path = None
                            st.rerun()
                    
                    with col_action2:
                        if st.button("💬 Ask AI for Feedback", use_container_width=True):
                            st.session_state.switch_to_chat = True
                            st.rerun()

# ==========================================
# TAB 3: AI CHAT
# ==========================================
with tab3:
    st.header("💬 Chat with Your AI Coach")
    
    st.markdown("""
    <div class="info-box">
    💡 <b>Ask me anything!</b><br>
    • "What exercises should I do for PCOS?"<br>
    • "How can I improve my plank form?"<br>
    • "What should I eat today?"<br>
    • "Can you explain the benefits of Tree Pose?"
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <b>You:</b> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f3e5f5; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <b>🤖 AI Coach:</b> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Type your message...", key="chat_input", placeholder="Ask me anything about PCOS exercises...")
    
    col_send, col_clear = st.columns([3, 1])
    
    with col_send:
        if st.button("📤 Send", use_container_width=True, type="primary"):
            if user_input:
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                with st.spinner("🤖 AI is thinking..."):
                    response = chat_with_ai(user_input)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
    
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ==========================================
# TAB 4: HISTORY
# ==========================================
with tab4:
    st.header("📊 Your Exercise & Chat History")
    
    history_tab1, history_tab2 = st.tabs(["🏋️ Exercise History", "💬 Chat History"])
    
    with history_tab1:
        if st.session_state.exercise_history:
            st.subheader(f"Total Exercises: {len(st.session_state.exercise_history)}")
            
            for idx, record in enumerate(reversed(st.session_state.exercise_history)):
                with st.expander(f"📅 {record['timestamp']} - {record['target_pose']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Target Exercise", record['target_pose'])
                    with col2:
                        st.metric("Detected Exercise", record['detected_pose'])
                    with col3:
                        match_status = "✅ Match" if record['target_pose'] == record['detected_pose'] else "❌ No Match"
                        st.metric("Result", match_status)
                    
                    st.progress(record['accuracy'] / 100)
                    st.caption(f"Accuracy: {record['accuracy']:.1f}% | Confidence: {record['confidence']*100:.1f}%")
        else:
            st.info("No exercise history yet. Upload and analyze a video to get started!")
    
    with history_tab2:
        if st.session_state.chat_history:
            st.subheader(f"Total Messages: {len(st.session_state.chat_history)}")
            
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 AI:** {message['content']}")
                st.markdown("---")
        else:
            st.info("No chat history yet. Start a conversation with the AI coach!")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🎯 About FlexiFit AI")
    
    st.markdown("""
    **FlexiFit AI** is your personal PCOS/PCOD exercise coach powered by:
    
    - 🤖 **MediaPipe** - Pose detection
    - 🧠 **Custom ML Model** - 92% accuracy
    - 🔍 **Elastic Search** - Exercise database
    - 💬 **Vertex AI Gemini** - AI coaching
    
    ---
    
    **✨ Features:**
    - Real-time pose detection
    - Video analysis with visual feedback
    - AI-powered coaching
    - Exercise history tracking
    - PCOS/PCOD-specific exercises
    
    ---
    
    **🏆 Built for Google Cloud Hackathon**
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All History", use_container_width=True):
        st.session_state.exercise_history = []
        st.session_state.chat_history = []
        st.success("History cleared!")
        st.rerun()
    
    st.markdown("---")

    st.caption("Made with ❤️ for women with PCOS/PCOD")

