# SClusterFormer RGB Plant Disease Classification - Memory Optimized for Kaggle

## 📦 Setup and Installation

import os
import sys
import subprocess
import gc
import torch

print("🚀 SClusterFormer RGB (Memory Optimized)")
print("="*40)

# Force CPU if GPU memory is limited
device = "cpu"  # Use CPU to avoid GPU memory issues
print(f"Using device: {device}")

# Install minimal packages
print("📦 Installing packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", "-f", "https://download.pytorch.org/whl/torch_stable.html", "--quiet"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "einops", "scikit-learn", "--quiet"])

## 📥 Clone Repository

print("📥 Cloning repository...")
if os.path.exists("plant_disease_classification"):
    import shutil
    shutil.rmtree("plant_disease_classification")

subprocess.check_call(["git", "clone", "https://github.com/rajanikant04/plant_disease_classification.git", "--quiet"])
os.chdir("plant_disease_classification")

## 🔧 Ultra-Minimal Configuration

# Set dataset path
DATASET_PATH = "/kaggle/input/apple-disease-dataset"

# Auto-detect available datasets
if os.path.exists("/kaggle/input"):
    datasets = os.listdir("/kaggle/input")
    print("Available datasets:", datasets)
    for item in datasets:
        if any(keyword in item.lower() for keyword in ['plant', 'disease', 'apple', 'leaf']):
            DATASET_PATH = f"/kaggle/input/{item}"
            print(f"Auto-selected: {DATASET_PATH}")
            break

print(f"Using dataset: {DATASET_PATH}")

# Create extreme memory-saving config
config_content = f'''# Extreme memory optimization for Kaggle
DATASET_PATH = "{DATASET_PATH}"
USE_VALIDATION_SPLIT = True
TRAIN_RATIO = 0.9  # Use more data for training, less for validation
RUN_TIMES = 1
IMG_SIZE = 64      # Very small images
BATCH_SIZE = 4     # Minimal batch size
EPOCHS = 5         # Quick training
LEARNING_RATE = 0.01  # Higher LR for faster convergence
INPUT_CHANNELS = 3
CUDA_DEVICES = []  # Force CPU usage
MODEL_CONFIG = {{
    "num_stages": 1,     # Single stage
    "n_groups": [2],     # Minimal groups
    "embed_dims": [32],  # Very small dimensions
    "num_heads": [1],    # Single attention head
    "mlp_ratios": [1],   # No MLP expansion
    "depths": [1],       # Single layer
}}
'''

with open('kaggle_config.py', 'w') as f:
    f.write(config_content)

print("✅ Ultra-minimal config created")

## 🚀 Training with Memory Management

# Clear any existing variables
gc.collect()

# Set memory-efficient environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['OMP_NUM_THREADS'] = '2'  # Limit CPU threads

print("🚀 Starting memory-optimized training...")

try:
    # Import and run directly to avoid subprocess overhead
    sys.path.insert(0, os.getcwd())
    
    # Set torch to use minimal memory
    torch.set_num_threads(2)
    
    # Import main module
    import main
    
    print("Running main.py with minimal memory usage...")
    result = main.run_experiment()
    
    if result is not None:
        print("✅ Training completed successfully!")
    else:
        print("⚠️ Training completed with warnings")
        
except Exception as e:
    print(f"❌ Training error: {e}")
    import traceback
    traceback.print_exc()

# Clean up
gc.collect()
print("🏁 Done!")