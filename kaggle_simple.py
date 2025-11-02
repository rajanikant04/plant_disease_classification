#!/usr/bin/env python
"""
Simple Kaggle runner for SClusterFormer RGB Plant Disease Classification
Just clone repo, set dataset path, and run main.py
"""

import os
import sys
import subprocess

print("🚀 SClusterFormer RGB Plant Disease Classification")
print("="*50)

# Install packages
print("📦 Installing packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "scikit-learn", "matplotlib"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "einops", "tqdm"])

# Clone repository
print("📥 Cloning repository...")
if os.path.exists("plant_disease_classification"):
    import shutil
    shutil.rmtree("plant_disease_classification")

subprocess.check_call(["git", "clone", "https://github.com/rajanikant04/plant_disease_classification.git"])
os.chdir("plant_disease_classification")

# ==========================================
# 🔧 SET YOUR DATASET PATH HERE
# ==========================================
DATASET_PATH = "/kaggle/input/apple-disease-dataset"
# Change this to match your Kaggle dataset path

print(f"📁 Dataset: {DATASET_PATH}")

# Create simple config override
with open('kaggle_config.py', 'w') as f:
    f.write(f'DATASET_PATH = "{DATASET_PATH}"\n')
    f.write('RUN_TIMES = 2\n')  # Reduced for Kaggle
    f.write('EPOCHS = 30\n')    # Reduced for Kaggle

# Run main.py
print("🚀 Running training...")
subprocess.check_call([sys.executable, "main.py"])

print("✅ Done!")