"""
Configuration file for Plant Disease Classification using SClusterFormer

🔧 MODIFY THESE SETTINGS FOR YOUR DATASET
"""

# ========================================
# 📁 DATASET CONFIGURATION
# ========================================

# Main dataset path - UPDATE THIS for your dataset
DATASET_PATH = "/kaggle/input/apple-disease-dataset"

# Alternative dataset paths (uncomment and modify as needed):
# DATASET_PATH = "/kaggle/input/plant-village-dataset"
# DATASET_PATH = "/kaggle/input/plantdoc-dataset" 
# DATASET_PATH = "/kaggle/input/new-plant-diseases-dataset"
# DATASET_PATH = "data/plant_disease"  # For local testing

# ========================================
# 📊 DATA SPLITTING CONFIGURATION
# ========================================

# Train/Validation splitting options
USE_VALIDATION_SPLIT = True     # If True: splits train folder into train/val
                               # If False: uses train folder as train, test folder as val

TRAIN_RATIO = 0.8              # Ratio for train/val split (0.8 = 80% train, 20% val)

# ========================================
# 🔧 TRAINING CONFIGURATION
# ========================================

# Training parameters
RUN_TIMES = 5          # Number of training runs (for statistical significance)
IMG_SIZE = 224         # Image size (224x224 is standard)
BATCH_SIZE = 32        # Batch size (adjust based on your GPU memory)
EPOCHS = 100           # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate
INPUT_CHANNELS = 3     # RGB channels

# GPU configuration
CUDA_DEVICES = [0]     # GPU devices to use [0] for single GPU

# ========================================
# 🎯 MODEL CONFIGURATION  
# ========================================

# SClusterFormer architecture parameters
MODEL_CONFIG = {
    'num_stages': 3,
    'n_groups': [16, 16, 16],       # Grouping for efficient computation
    'embed_dims': [256, 128, 64],   # Feature dimensions for each stage
    'num_heads': [8, 4, 2],         # Attention heads for each stage
    'mlp_ratios': [4, 4, 4],        # MLP expansion ratios
    'depths': [2, 2, 2],            # Number of blocks in each stage
}

# ========================================
# 📋 EXPERIMENT NAMING
# ========================================

# Experiment name (for saving results)
EXPERIMENT_NAME = "SClusterFormer_PlantDisease"

# Results directory
RESULTS_DIR = "results"

# ========================================
# 🚀 QUICK SETUP EXAMPLES
# ========================================

"""
EXAMPLE 1: Kaggle Apple Disease Dataset
DATASET_PATH = "/kaggle/input/apple-disease-dataset"
USE_VALIDATION_SPLIT = True
TRAIN_RATIO = 0.8

EXAMPLE 2: PlantVillage Dataset  
DATASET_PATH = "/kaggle/input/plant-village-dataset"
USE_VALIDATION_SPLIT = True
TRAIN_RATIO = 0.85

EXAMPLE 3: Custom Local Dataset
DATASET_PATH = "data/my_plant_dataset"
USE_VALIDATION_SPLIT = True
TRAIN_RATIO = 0.75

EXAMPLE 4: Pre-split Dataset (has separate train/test folders)
DATASET_PATH = "/kaggle/input/presplit-plant-dataset"
USE_VALIDATION_SPLIT = False  # Use original train/test split
"""

# ========================================
# 📖 DATASET STRUCTURE REQUIREMENTS
# ========================================

"""
Your dataset should follow this structure:

Option 1: Single directory (will be split automatically)
dataset/
    class1/
        img1.jpg
        img2.jpg
    class2/
        img3.jpg
        img4.jpg

Option 2: Pre-split structure
dataset/
    datasets/  # (optional wrapper)
        train/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                img4.jpg
        test/
            class1/
                img5.jpg
            class2/
                img6.jpg

Supported image formats: .jpg, .jpeg, .png
"""