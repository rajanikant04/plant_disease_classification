import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

# Model imports with fallback logic
try:
    from models.SClusterFormer import SClusterFormer
    MODEL_TYPE = "SClusterFormer"
    print("✅ Using full SClusterFormer model")
except ImportError as e:
    print(f"⚠️ Could not import SClusterFormer: {e}")
    try:
        from models.MinimalSClusterFormer import MinimalSClusterFormer as SClusterFormer
        MODEL_TYPE = "MinimalSClusterFormer"
        print("✅ Using MinimalSClusterFormer fallback")
    except ImportError as e2:
        print(f"❌ Could not import fallback model: {e2}")
        print("🔧 Creating emergency minimal model...")
        
        import torch.nn as nn
        
        class EmergencyModel(nn.Module):
            def __init__(self, input_channels=3, num_classes=4, **kwargs):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(input_channels, 16, 7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(16, num_classes)
                )
            
            def forward(self, x):
                return self.backbone(x)
        
        SClusterFormer = EmergencyModel
        MODEL_TYPE = "EmergencyModel"
        print("✅ Using emergency minimal model")

from rgb_data_loader import load_kaggle_plant_disease_data, load_plant_disease_data
from Loop_RGB_train import loop_train_test_rgb

# Import configuration
try:
    # Try Kaggle config first, then regular config
    try:
        from kaggle_config import *
        print("✅ Loaded Kaggle configuration from kaggle_config.py")
        # Set defaults for missing values
        globals().setdefault('USE_VALIDATION_SPLIT', True)
        globals().setdefault('TRAIN_RATIO', 0.8)
        globals().setdefault('IMG_SIZE', 224)
        globals().setdefault('BATCH_SIZE', 32)
        globals().setdefault('LEARNING_RATE', 0.001)
        globals().setdefault('INPUT_CHANNELS', 3)
        globals().setdefault('CUDA_DEVICES', [0])
        globals().setdefault('MODEL_CONFIG', {
            'num_stages': 3,
            'n_groups': [16, 16, 16],
            'embed_dims': [256, 128, 64],
            'num_heads': [8, 4, 2],
            'mlp_ratios': [4, 4, 4],
            'depths': [2, 2, 2],
        })
    except ImportError:
        from config import *
        print("✅ Loaded configuration from config.py")
    
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"🔄 Validation split: {'Enabled' if USE_VALIDATION_SPLIT else 'Disabled'}")
except ImportError:
    print("⚠️  No config file found, using default configuration")
    # Default configuration
    DATASET_PATH = "/kaggle/input/apple-disease-dataset"
    USE_VALIDATION_SPLIT = True
    TRAIN_RATIO = 0.8
    RUN_TIMES = 5
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    INPUT_CHANNELS = 3
    CUDA_DEVICES = [0]
    MODEL_CONFIG = {
        'num_stages': 3,
        'n_groups': [16, 16, 16],
        'embed_dims': [256, 128, 64],
        'num_heads': [8, 4, 2],
        'mlp_ratios': [4, 4, 4],
        'depths': [2, 2, 2],
    }

# Remove abundant output
warnings.filterwarnings('ignore')


# All training functions are now in Loop_RGB_train.py


def run_experiment():
    """Main experiment function using Loop_RGB_train"""
    print("SClusterFormer RGB Plant Disease Classification")
    print("="*60)
    print(f"🤖 Model Type: {MODEL_TYPE}")
    
    # Adjust configuration based on model type
    global MODEL_CONFIG, IMG_SIZE, BATCH_SIZE, EPOCHS
    
    if MODEL_TYPE in ["MinimalSClusterFormer", "EmergencyModel"]:
        print("🔧 Applying fallback model optimizations...")
        # Use more conservative settings for fallback models
        IMG_SIZE = min(IMG_SIZE, 64)  # Cap image size
        BATCH_SIZE = min(BATCH_SIZE, 8)  # Reduce batch size
        EPOCHS = min(EPOCHS, 20)  # Reduce epochs
        
        # Simplify model configuration
        MODEL_CONFIG = {
            'num_stages': 1,
            'n_groups': [2],
            'embed_dims': [16],
            'num_heads': [1],
            'mlp_ratios': [1],
            'depths': [1],
        }
        
        print(f"📊 Adjusted config: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
    
    print(f"🔧 Final model config: {MODEL_CONFIG}")
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset not found! Please update DATASET_PATH in config")
        print(f"Current path: {DATASET_PATH}")
        print("📝 Available paths in /kaggle/input:")
        try:
            if os.path.exists("/kaggle/input"):
                for item in os.listdir("/kaggle/input"):
                    print(f"  - /kaggle/input/{item}")
        except:
            pass
        print("💡 Update DATASET_PATH to match one of the above paths")
        return None
    
    print("✅ Dataset path found!")
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"🔄 Validation split: {'Enabled' if USE_VALIDATION_SPLIT else 'Disabled'}")
    
    # Prepare hyperparameters for Loop_RGB_train
    hyperparameters = {
        'img_size': IMG_SIZE,
        'input_channels': INPUT_CHANNELS,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'cuda': CUDA_DEVICES,
        'run_times': RUN_TIMES,
        'train_ratio': TRAIN_RATIO,
        'data_path': DATASET_PATH,
        'use_validation_split': USE_VALIDATION_SPLIT,
        'model_config': MODEL_CONFIG
    }
    
    print(f"🚀 Starting training with {RUN_TIMES} runs...")
    print(f"📊 Model config: {MODEL_CONFIG}")
    
    # Call the training loop from Loop_RGB_train.py
    results = loop_train_test_rgb(hyperparameters)
    
    print("✅ Training completed successfully!")
    return results


if __name__ == '__main__':
    run_experiment()
    print(f"\n✅ Experiment completed at {time.asctime(time.localtime())}")