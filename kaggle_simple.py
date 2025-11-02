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

# Create Kaggle-optimized config override
with open('kaggle_config.py', 'w') as f:
    f.write(f'DATASET_PATH = "{DATASET_PATH}"\n')
    f.write('USE_VALIDATION_SPLIT = True\n')
    f.write('TRAIN_RATIO = 0.8\n')
    f.write('RUN_TIMES = 1\n')  # Single run for Kaggle
    f.write('IMG_SIZE = 224\n')
    f.write('BATCH_SIZE = 16\n')  # Reduced batch size for memory
    f.write('EPOCHS = 20\n')     # Reduced epochs for time limit
    f.write('LEARNING_RATE = 0.001\n')
    f.write('INPUT_CHANNELS = 3\n')
    f.write('CUDA_DEVICES = [0]\n')
    f.write('MODEL_CONFIG = {\n')
    f.write('    "num_stages": 2,\n')      # Reduced model complexity
    f.write('    "n_groups": [8, 8],\n')   # Smaller groups
    f.write('    "embed_dims": [128, 64],\n')  # Smaller dimensions
    f.write('    "num_heads": [4, 2],\n')   # Fewer attention heads
    f.write('    "mlp_ratios": [2, 2],\n') # Smaller MLP ratios
    f.write('    "depths": [1, 1],\n')     # Fewer transformer blocks
    f.write('}\n')

# Check available memory
print("💾 Checking system resources...")
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"Available Memory: {memory.available / (1024**3):.1f} GB")
except:
    print("Memory info not available")

# Run main.py with error handling
print("🚀 Running training...")
try:
    result = subprocess.run([sys.executable, "main.py"], 
                          capture_output=True, text=True, timeout=3600)  # 1 hour timeout
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        print(result.stdout)
    else:
        print("❌ Training failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
except subprocess.TimeoutExpired:
    print("⏰ Training timed out (>1 hour)")
except Exception as e:
    print(f"❌ Error running training: {e}")

print("🏁 Kaggle run finished!")