![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

# ⚾ PitchIQ - AI Baseball Pitch Prediction Service

> **Deep Learning 기반 다음 구종 예측 서비스**
>
> 이전 투구 패턴과 경기 상황을 분석하여 투수가 다음에 던질 구종을 예측하는 AI 시스템

---

# 📌 Project Overview

PitchIQ는 MLB Statcast 데이터를 활용하여 **다음 투구 구종을 예측**하는 딥러닝 프로젝트입니다.

기존 야구 중계에서는 투수가 공을 던진 이후 구종과 구속이 표시됩니다.

PitchIQ는 여기에 더해

> **"다음에는 어떤 공을 던질까?"**

를 AI가 예측하여 보다 몰입감 있는 경기 관람 경험을 제공하는 것을 목표로 합니다.

---

# 🎯 Features

- ⚾ MLB Statcast 데이터 기반 학습
- 🧠 BiLSTM 기반 다음 구종 예측
- 👤 투수 / 타자 Embedding 적용
- 🎥 YOLOv8 기반 야구공 객체 탐지
- 📊 예측 결과 시각화
- 🌐 Streamlit 웹 서비스

---

# 🏗 Architecture

```
Video
   │
   ▼
YOLOv8 Ball Detection
   │
   ▼
Pitch Trajectory
   │
   ▼
Feature Extraction
   │
   ▼
BiLSTM Pitch Prediction
   │
   ▼
Pitch Probability
   │
   ▼
Streamlit Dashboard
```

---

# 🛠 Tech Stack

| 분야 | 기술 |
|------|------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Object Detection | YOLOv8 |
| Data | MLB Statcast (pybaseball) |
| Visualization | Matplotlib |
| Dashboard | Streamlit |

---

# 📂 Project Structure

```
baseball-pitch-predictor/

├── data/
│   └── raw/
│
├── models/
│   ├── pitch_predictor.h5
│   └── baseball_detector/
│
├── notebooks/
│   └── eda.py
│
├── src/
│   ├── data_collector.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── evaluate.py
│   ├── train_yolo.py
│   └── yolo_detector.py
│
├── streamlit_app/
│
└── README.md
```

---

# 🤖 Deep Learning Model

### Input

- Previous 5 pitches
- Pitcher ID
- Batter ID
- Game Context

### Model

- Bidirectional LSTM
- Embedding Layer
- Dense Layer
- Softmax Classifier

### Output

8 Pitch Classes

- FF
- SI
- FC
- SL
- CU
- CH
- FS
- OTHER

---

# 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| Random Guess | 12.5% |
| Initial LSTM | 39% |
| Final BiLSTM | **46.14%** |

---

# ⚙ Pipeline

```
Statcast Data
        │
        ▼
Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
BiLSTM Training
        │
        ▼
Prediction
        │
        ▼
Streamlit Service
```

---

# 🚀 Future Work

- 실시간 경기 영상 분석
- Pitch Tracking 자동화
- MLB → KBO 확장
- 선수 맞춤 분석
- 모바일 서비스 개발

---

# 👨‍💻 Developer

**박지수**

AI Deep Learning Project

2026
