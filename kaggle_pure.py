# SClusterFormer RGB - Pure Kaggle Environment (No Installations)

import os
import sys
import subprocess
import gc

print("🚀 SClusterFormer RGB (Pure Kaggle Environment)")
print("="*50)

# NO package installations - use only Kaggle's built-in packages
print("📦 Using Kaggle's pre-installed packages...")

# Clone repository
print("📥 Cloning repository...")
if os.path.exists("plant_disease_classification"):
    import shutil
    shutil.rmtree("plant_disease_classification")

subprocess.check_call(["git", "clone", "https://github.com/rajanikant04/plant_disease_classification.git", "--quiet"])
os.chdir("plant_disease_classification")

# Auto-detect dataset
DATASET_PATH = "/kaggle/input/apple-disease-dataset"

if os.path.exists("/kaggle/input"):
    datasets = os.listdir("/kaggle/input")
    print(f"Available datasets: {datasets}")
    for item in datasets:
        if any(keyword in item.lower() for keyword in ['plant', 'disease', 'apple', 'leaf']):
            DATASET_PATH = f"/kaggle/input/{item}"
            print(f"Auto-selected: {DATASET_PATH}")
            break

print(f"Using dataset: {DATASET_PATH}")

# Create ultra-simple configuration for your dataset structure
config_content = f'''# Simple Kaggle configuration
DATASET_PATH = "{DATASET_PATH}/datasets"
USE_VALIDATION_SPLIT = False  # Use pre-split train/test folders
TRAIN_RATIO = 0.8
RUN_TIMES = 1
IMG_SIZE = 112     # GPU-optimized size
BATCH_SIZE = 4     # Very small batch for GPU memory safety
EPOCHS = 20        # Good training
LEARNING_RATE = 0.001
INPUT_CHANNELS = 3
CUDA_DEVICES = [0]  # Use GPU device 0
MODEL_CONFIG = {{
    "num_stages": 2,      # GPU-optimized stages
    "n_groups": [4, 4],   # Small groups for memory
    "embed_dims": [64, 32], # Small dimensions for GPU memory
    "num_heads": [2, 1],  # Minimal heads
    "mlp_ratios": [2, 2], # Standard ratios
    "depths": [1, 1],     # Single layer per stage
}}
'''

with open('kaggle_config.py', 'w') as f:
    f.write(config_content)

print("✅ Configuration created for pre-split dataset")
print("📊 Config: 112x112 images, batch=4, 20 epochs, GPU-optimized, using train/test folders")

# Optimize for Kaggle GPU usage
print("🔧 Setting up GPU memory management...")
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:64,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA for better performance
print("🖥️ Optimized for Kaggle GPU usage with aggressive memory management")

# Import required modules with error handling
print("📚 Importing modules...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    
    # Check GPU availability and optimize memory
    if torch.cuda.is_available():
        print(f"🖥️ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()  # Clear GPU cache
    else:
        print("⚠️ No GPU available, will use CPU")
    
    import torchvision
    print(f"✅ TorchVision {torchvision.__version__}")
    
    # Check for einops (install if missing)
    try:
        import einops
        print(f"✅ einops available")
    except ImportError:
        print("📦 Installing einops...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "einops", "--quiet"])
        print("✅ einops installed")
        
    print("🔧 PyTorch configured for GPU usage with memory optimization")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Run training
print("🚀 Starting training...")
try:
    # Run main.py using subprocess for better isolation
    result = subprocess.run(
        [sys.executable, "main.py"], 
        capture_output=True, 
        text=True, 
        timeout=3600,  # 1 hour timeout
        env=dict(os.environ, PYTHONPATH=os.getcwd())
    )
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        print("\n📊 Training Output:")
        # Show last 20 lines of output
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines[-20:]:
            if line.strip():
                print(line)
    else:
        print("❌ Training failed!")
        print("\n🔍 Error Output:")
        print(result.stderr[-2000:])  # Last 2000 chars of error
        
except subprocess.TimeoutExpired:
    print("⏰ Training timeout (1 hour exceeded)")
except Exception as e:
    print(f"❌ Execution error: {e}")

print("\n🏁 Kaggle run finished!")