
## What is FlexiFit AI?

**FlexiFit AI** is a compassionate health and fitness platform designed *by a woman, for women.*  
It’s a safe, inclusive space to open up about health challenges, find exercises tailored to your condition, and perform them correctly with **real-time AI guidance**.

Our AI checks your **pose and angle** live to ensure safety and correctness.  
If it detects you’re struggling, it automatically switches to a **modified version** of the pose helping you stay consistent and motivated instead of giving up.  
You can also **track your progress**, ask **AI health questions**, and maintain a private **90-day health record** all safely and personally.

---

## The Story Behind FlexiFit AI

> *“I built FlexiFit AI because I’ve lived the problem myself.”*

I’ve been dealing with **PCOS**, **fatigue**, **weight gain**, and **months of missed periods**.  
Due to financial struggles, I couldn’t visit a doctor regularly.  
So, I turned to **YouTube** for workouts.

One day, I tried the **Cobra Pose** it looked easy, so I thought I was doing it right.  
Later, during my search for PCOS-friendly exercises, I found another video explaining that the **arms shouldn’t be fully straight the elbows should be slightly bent**.  
That’s when I realized I had been doing the pose wrong all along.  

It wasn’t just that one many of my poses were incorrect. That caused fatigue, pain, and no results.  
YouTube could show me exercises but couldn’t *correct* my posture or *tell me* if I was doing it wrong.

Even when I asked **ChatGPT** for diet help (which gave great suggestions within my budget), it couldn’t **track my progress** or **monitor my exercises** and I lost motivation.  
Other exercise apps required **premium subscriptions** for nutrition advice or pose correction.

So when this **hackathon** came up, I decided to build something that truly *understands women like me.*  
That’s how **FlexiFit AI** was born an affordable, empathetic AI that doesn’t just talk it *sees, learns, guides, and remembers.*

---

## Why FlexiFit AI?

| Problem | YouTube | ChatGPT | FlexiFit AI |
|----------|----------|----------|-------------|
| **Real-time form correction** | No | Text only | Yes — Live pose & angle detection |
| **Adaptive difficulty** | No | Doesn't adapt | Yes — Auto-switch to easier poses |
| **Remembers your history** | No | No memory | Yes — 90-day health tracking |
| **Culturally aware diet** | Western foods | Generic advice | Yes — Indian, budget-friendly meals |
| **Prevents injury** | No feedback | No vision | Yes — Red warning for wrong angles |
| **Motivation system** | Watch & forget | No gamification | Yes — Round system, progress tracking |
| **Doctor-ready reports** | No data | No reports | Yes — Export 90-day summary |
| **Language support** | English only | Multi-language (text) | Coming soon — multilingual chat & voice feedback |

---

## Current Setup

### Streamlit App

#### Why Streamlit?
Streamlit was chosen to **rapidly prototype AI + UI workflows** for the hackathon.  
It allows testing of **pose detection**, **video uploads**, and **health tracking** efficiently even though real-time streaming isn’t supported natively.  
If the app lags or reruns unexpectedly, **refresh the Streamlit page** to restore performance.

#### Features

- **Workout Tracker:**  
  Browse curated exercises like *beginner workouts*, *weight loss routines*, and *condition-based yoga.*  
  AI suggests the best ones for your goals and health inputs.

- **Analyze Video:**  
  Upload your workout video and get instant AI feedback.  
  - *Green lines:* Correct pose  
  - *Red lines:* Incorrect pose  
  - Download the annotated video for review.  
  

- **AI Health Chat:**  
  Ask anything related to your health, diet, or symptoms.  
  The AI provides **budget-conscious**, **culturally relevant** suggestions and remembers your (demo) history.

- **Health Tracker:**  
  Displays weight, symptom, and progress trends over 90 days.  
  *(Prototype data is fake the real version will use secure, consent-based storage.)*

- **Progress Stats:**  
  Displays your recent workouts and chat history.  
  Once the session ends or you refresh the page, your information resets for privacy.

---

### Live Webcam Mode

#### Why a Separate Module?
Streamlit cannot handle real-time pose estimation efficiently, so a **dedicated webcam version** using **OpenCV + MediaPipe** was built for smooth live feedback.

#### Features

- **Pose Detection:**  
  Real-time skeleton overlay:  
  Green = Correct posture  
  Red = Incorrect posture  

- **Round System:**  
  Each correct pose counts as one “round.”  
  Currently set to 2 rounds, 5-second duration, and 3-second rest.

- **Auto Switch Mode:**  
  After 10 failed frames, AI automatically switches to a **modified pose version.**  


- **Angle Visualization:**  
  Currently used for demonstration only real-time angle tracking is in progress.

#### Exercises Implemented
Prototype includes 5 demo exercises:
1. Downward Dog  
2. Warrior II  
3. Plank  
4. Standard Tree Pose  
5. Modified Tree Pose  

> Due to lighting and camera position issues, precise angle calculations are postponed for stability.  
> The current version visually demonstrates how angle-based correction will work.

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **AI / ML** | PyTorch, MediaPipe |
| **Backend / Database** | Elasticsearch |
| **Cloud & ML Platform** | Google Cloud Vertex AI |
| **Video & Image Processing** | OpenCV, Pillow |
| **Data Handling** | NumPy |
| **Future Frontend** | React Native (for mobile app) |

---

## Limitations (Prototype Stage)

| Area | Current State | Next Steps |
|------|----------------|------------|
| **Pose Angles** | Used for visualization only | Implement 3D landmark angle-based scoring |
| **Streamlit Live Feed** | Not real-time | Move to React Native + WebRTC |
| **Session Data** | Not saved; resets on refresh | Add persistent local storage with consent |
| **Demo Data** | Fake 90-day data | Replace with secure real user data |
| **Exercise Range** | Only 5 demo poses | Expand to 50+ exercises for specific conditions |
| **Language Support** | English only | Add Hindi, Tamil, Bengali & regional languages |
| **Speech Recognition** | Not yet implemented | Add live speech guidance (“Raise your arm”, “Hold your pose”) |
| **Auto Switch** | Triggers quickly | Add smooth threshold adjustment |
| **UI/UX** | Basic prototype | Add guided voice feedback & modern mobile UI |

---

## Future Scope

### Angle-Based AI Correction
Future versions will use **3D body landmark angles** (via MediaPipe + OpenCV) to:
- Detect incorrect joint alignment (e.g., bent knees, slouched back)
- Offer *real-time voice feedback* such as:  
  “Straighten your spine a little,” or  
  “Bend your elbow to 45°.”

---

### Speech Recognition & Voice Feedback
During live webcam workouts:
- The app will listen and respond, for example:  
  “FlexiFit, start Tree Pose,” → begins tracking.  
- It will speak back for motivation:  
  “Good job! Hold your posture.”  
- Progress and feedback will be stored automatically for reports and insights.

---

### Multilingual Support
Support for regional languages and intelligent voice assistants such as Hindi, Tamil, Bengali, and Marathi will be introduced.  
This ensures that every woman can interact with FlexiFit AI in the language she is most comfortable with because health should never have a language barrier.

---

### React Native Mobile App
Transform the current prototype into a full-fledged mobile application for seamless, real-time AI coaching and posture correction.

---

### Doctor Integration
Enable verified doctors and physiotherapists to review user summaries, provide professional feedback, and recommend personalized exercise plans.

---

### Community Support
Create a moderated and safe community space where women can share experiences, exchange advice, and motivate each other on their health journeys.

---

### Expanded Health Library
Include condition-specific workouts and wellness routines tailored for PCOS, thyroid imbalance, back pain, anxiety management, and fatigue recovery.

---

### Privacy by Design
All personal data will be encrypted, anonymized, and stored only with explicit user consent.  
FlexiFit AI is committed to ensuring safety, trust, and complete data privacy at every level.


---

## Our Vision

> In India, many women silently struggle with PCOS, fatigue, or pain unable to afford trainers or explain symptoms clearly to doctors.  
> FlexiFit AI gives them empathy, science, and a safe digital space to grow stronger physically and emotionally.


---
