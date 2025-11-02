# SClusterFormer RGB - Kaggle Compatible Version

import os
import sys
import subprocess
import gc

print("🚀 SClusterFormer RGB (Kaggle Compatible)")
print("="*40)

# Use Kaggle's built-in packages (no installation needed)
print("📦 Using Kaggle's built-in PyTorch...")

# Install only minimal additional packages that don't conflict
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "einops", "--quiet"])
    print("✅ Installed einops")
except:
    print("⚠️ einops installation failed, using fallback")

# Clone repository
print("📥 Cloning repository...")
if os.path.exists("plant_disease_classification"):
    import shutil
    shutil.rmtree("plant_disease_classification")

subprocess.check_call(["git", "clone", "https://github.com/rajanikant04/plant_disease_classification.git", "--quiet"])
os.chdir("plant_disease_classification")

# Dataset configuration
DATASET_PATH = "/kaggle/input/apple-disease-dataset"

# Auto-detect datasets
if os.path.exists("/kaggle/input"):
    datasets = os.listdir("/kaggle/input")
    print("Available datasets:", datasets)
    for item in datasets:
        if any(keyword in item.lower() for keyword in ['plant', 'disease', 'apple', 'leaf']):
            DATASET_PATH = f"/kaggle/input/{item}"
            print(f"Auto-selected: {DATASET_PATH}")
            break

print(f"Using dataset: {DATASET_PATH}")

# Create config optimized for your dataset structure
config_content = f'''# Kaggle configuration for apple-disease-dataset
DATASET_PATH = "{DATASET_PATH}/datasets"  # Point to datasets folder
USE_VALIDATION_SPLIT = False  # Dataset already has train/test split
TRAIN_RATIO = 0.8  # Not used since pre-split
RUN_TIMES = 1
IMG_SIZE = 128     # Memory-safe size
BATCH_SIZE = 16    # Reasonable batch size
EPOCHS = 15        # Good training length
LEARNING_RATE = 0.001
INPUT_CHANNELS = 3
CUDA_DEVICES = [0] if __import__('torch').cuda.is_available() else []
MODEL_CONFIG = {{
    "num_stages": 2,      # Good complexity
    "n_groups": [8, 8],   # Adequate groups  
    "embed_dims": [128, 64], # Good dimensions
    "num_heads": [4, 2],  # Reasonable heads
    "mlp_ratios": [2, 2], # Standard ratios
    "depths": [2, 1],     # Good depth
}}
'''

with open('kaggle_config.py', 'w') as f:
    f.write(config_content)

print("✅ Memory-safe config created")

# Memory optimizations
import torch
torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = '4'

print("🚀 Running training...")

try:
    # Run with subprocess for better isolation
    result = subprocess.run(
        [sys.executable, "main.py"], 
        capture_output=True, 
        text=True, 
        timeout=2400,  # 40 minutes
        cwd=os.getcwd()
    )
    
    if result.returncode == 0:
        print("✅ SUCCESS! Training completed")
        # Show key results
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if any(keyword in line.lower() for keyword in ['accuracy', 'loss', 'completed', 'results']):
                print(f"📊 {line}")
    else:
        print("❌ Training failed")
        print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
        
except subprocess.TimeoutExpired:
    print("⏰ Training timeout (40 min)")
except Exception as e:
    print(f"❌ Error: {e}")

print("🏁 Done!")