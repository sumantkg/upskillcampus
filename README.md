# upskillcampus
# 🌾 Crop and Weed Detection + ✈️ Turbofan Engine RUL Prediction

## 📖 Project Overview

This project consists of two machine learning-based applications:

1. **Crop and Weed Detection System**
2. **Remaining Useful Life (RUL) Prediction for Turbofan Engines**

The goal is to apply computer vision and predictive modeling techniques to solve real-world agricultural and industrial problems.

---

# 🌱 1. Crop and Weed Detection System

## 🎯 Objective
To automatically detect and classify crops and weeds from agricultural field images using deep learning techniques.

## 🧠 Technologies Used
- Python
- OpenCV
- TensorFlow / PyTorch (as applicable)
- YOLO / CNN-based models

## ⚙️ Working
- Input: Field images or video frames
- Process:
  - Image preprocessing
  - Object detection model identifies crops and weeds
- Output:
  - Bounding boxes around crops and weeds
  - Classification labels

## 🚀 Features
- Real-time detection capability
- High accuracy classification
- Helps in precision agriculture
- Reduces manual labor

---

# ✈️ 2. Turbofan Engine RUL Prediction

## 🎯 Objective
To predict the Remaining Useful Life (RUL) of turbofan engines using sensor data and machine learning models.

## 🧠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- LSTM / Regression models

## ⚙️ Working
- Input: Time-series sensor data from engines
- Process:
  - Data preprocessing and normalization
  - Feature extraction
  - Model training
- Output:
  - Predicted remaining operational cycles before failure

## 🚀 Features
- Predictive maintenance support
- Reduces unexpected engine failure
- Improves safety and cost efficiency

---

# 📊 Dataset Information

### Crop & Weed Detection:
- Agricultural image dataset (custom / Kaggle / UAV dataset)

### RUL Prediction:
- NASA CMAPSS Turbofan Engine dataset

---

# 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/upskillcampus.git

# Navigate to project folder
cd upskillcampus

# Install dependencies
pip install -r requirements.txt
