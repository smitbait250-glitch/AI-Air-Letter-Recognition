# ✋ AI Air Letter Recognition

An AI-based computer vision project that recognizes **letters drawn in the air using hand movements**.

The project uses a webcam to track hand landmarks, extract the movement of the user's hand, process the drawn letter, and use a trained deep-learning model to predict the corresponding alphabet letter.

---

## 📌 Project Overview

**AI Air Letter Recognition** allows users to write letters in the air without touching a keyboard, screen, or physical surface.

The system uses:

- 🖐️ Hand tracking for detecting hand movement
- 📷 Webcam for real-time input
- 🧠 TensorFlow for deep-learning based letter classification
- 👁️ MediaPipe for hand landmark detection
- 🔢 OpenCV for image processing
- 📚 EMNIST Letters dataset for model training

The main goal of the project is to demonstrate how **Computer Vision + Machine Learning** can be combined to create a touchless letter recognition system.

---

## ✨ Features

- 🎥 Real-time webcam input
- 🖐️ Real-time hand tracking
- ✍️ Air-writing letter detection
- 🧠 AI-based letter classification
- 🔤 Recognition of alphabet letters
- ⚡ Real-time prediction
- 🖥️ Runs locally on a computer
- 🚫 No physical writing surface required

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| TensorFlow | Machine learning model |
| MediaPipe | Hand landmark detection |
| OpenCV | Webcam and image processing |
| NumPy | Numerical operations |
| TensorFlow Datasets | EMNIST dataset |
| EMNIST Letters | Training dataset |

---

## 🧠 How It Works

The system follows these main steps:

```text
Webcam
   ↓
Hand Detection
   ↓
Hand Landmark Tracking
   ↓
Air-Writing Movement
   ↓
Image Preprocessing
   ↓
28 × 28 Input
   ↓
CNN Model
   ↓
Letter Prediction