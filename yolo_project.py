# ==========================================
# Crop & Weed Detection using YOLOv8
# ==========================================

from ultralytics import YOLO
import os

# ------------------------------
# 1. CHECK FILES
# ------------------------------
print("Checking dataset structure...")

if not os.path.exists("data.yaml"):
    print("❌ data.yaml not found!")
    exit()

if not os.path.exists("dataset/images/train"):
    print("❌ Training images folder not found!")
    exit()

if not os.path.exists("dataset/labels/train"):
    print("❌ Training labels folder not found!")
    exit()

print("✅ Dataset structure looks correct!")

# ------------------------------
# 2. LOAD MODEL
# ------------------------------
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")   # lightweight model

# ------------------------------
# 3. TRAIN MODEL
# ------------------------------
print("Starting training...")

model.train(
    data="data.yaml",
    epochs=5,
    imgsz=512,
    batch=8,
    name="crop_weed_model"
)

print("✅ Training completed!")

# ------------------------------
# 4. TEST / PREDICTION
# ------------------------------
print("Running prediction on validation image...")

# Change this to any image in your val folder
test_image = "dataset/images/val"

results = model.predict(
    source=test_image,
    save=True,     # saves output
    show=True      # shows image
)

print("✅ Prediction completed!")

# ------------------------------
# 5. MODEL SAVE INFO
# ------------------------------
print("\n📁 Model saved at:")
print("runs/detect/crop_weed_model/weights/best.pt")

print("\n🎉 Project Completed Successfully!")