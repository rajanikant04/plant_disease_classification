#!/usr/bin/env python
"""
Kaggle-optimized runner for SClusterFormer RGB Plant Disease Classification
Ultra-lightweight version to avoid memory/timeout issues
"""

import os
import sys
import subprocess

print("🚀 SClusterFormer RGB (Kaggle Optimized)")
print("="*40)

# Quick package installation
print("📦 Installing essential packages...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.1", "torchvision==0.15.2", "--quiet"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "einops", "scikit-learn", "--quiet"])
    print("✅ Packages installed")
except Exception as e:
    print(f"⚠️ Package installation issue: {e}")

# Clone repo
print("📥 Cloning repository...")
if os.path.exists("plant_disease_classification"):
    import shutil
    shutil.rmtree("plant_disease_classification")

try:
    subprocess.check_call(["git", "clone", "https://github.com/rajanikant04/plant_disease_classification.git", "--quiet"])
    os.chdir("plant_disease_classification")
    print("✅ Repository cloned")
except Exception as e:
    print(f"❌ Clone failed: {e}")
    sys.exit(1)

# ==========================================
# 🔧 DATASET CONFIGURATION
# ==========================================
DATASET_PATH = "/kaggle/input/apple-disease-dataset"
print(f"📁 Target dataset: {DATASET_PATH}")

# Check what's actually available
if os.path.exists("/kaggle/input"):
    print("📋 Available datasets:")
    for item in os.listdir("/kaggle/input"):
        print(f"  - /kaggle/input/{item}")
        # Auto-detect plant disease datasets
        if any(keyword in item.lower() for keyword in ['plant', 'disease', 'leaf', 'apple']):
            DATASET_PATH = f"/kaggle/input/{item}"
            print(f"🎯 Auto-selected: {DATASET_PATH}")
            break

# Create ultra-lightweight config for your dataset
print("⚙️ Creating minimal config...")
config_content = f'''# Ultra-lightweight Kaggle config for apple-disease-dataset
DATASET_PATH = "{DATASET_PATH}/datasets"  # Point to datasets folder  
USE_VALIDATION_SPLIT = False  # Use pre-split train/test folders
TRAIN_RATIO = 0.8
RUN_TIMES = 1
IMG_SIZE = 96           # Very small image size (96x96)
BATCH_SIZE = 4          # Tiny batch size
EPOCHS = 10             # Quick training
LEARNING_RATE = 0.001
INPUT_CHANNELS = 3
CUDA_DEVICES = [0] if __import__('torch').cuda.is_available() else []
MODEL_CONFIG = {{
    "num_stages": 1,         # Single stage only
    "n_groups": [2],         # Minimal groups
    "embed_dims": [32],      # Very small dimensions
    "num_heads": [1],        # Single attention head
    "mlp_ratios": [1],       # No MLP expansion
    "depths": [1],           # Single layer
}}
'''

with open('kaggle_config.py', 'w') as f:
    f.write(config_content)

print("✅ Minimal config created")

# Run with maximum safety
print("🚀 Starting lightweight training...")
try:
    # Set memory-efficient environment
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    result = subprocess.run([sys.executable, "main.py"], 
                          capture_output=True, text=True, 
                          timeout=1800)  # 30 min timeout
    
    if result.returncode == 0:
        print("✅ SUCCESS! Training completed")
        # Show last few lines of output
        output_lines = result.stdout.split('\n')
        print("📊 Results:")
        for line in output_lines[-10:]:
            if line.strip():
                print(f"  {line}")
    else:
        print("❌ Training failed")
        print("Last error lines:")
        for line in result.stderr.split('\n')[-5:]:
            if line.strip():
                print(f"  ERROR: {line}")
                
except subprocess.TimeoutExpired:
    print("⏰ Training timeout (30 min exceeded)")
except Exception as e:
    print(f"❌ Execution error: {e}")

print("🏁 Kaggle run complete")