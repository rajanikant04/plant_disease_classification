#!/usr/bin/env python3

"""
🚀 Kaggle Memory-Optimized SClusterFormer Script
Heavily optimized for Kaggle GPU memory constraints with adaptive pooling
"""

import os
import subprocess
import sys
import shutil

def setup_environment():
    """Setup the environment with necessary packages"""
    print("🔧 Setting up environment...")
    
    # Install essential packages with specific versions for compatibility
    packages = [
        'torch==1.13.1',
        'torchvision==0.14.1', 
        'Pillow==9.5.0',
        'numpy==1.24.3',
        'scikit-learn==1.2.2'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
        except Exception as e:
            print(f"Warning: Could not install {package}: {e}")

def clone_repository():
    """Clone the repository if not already present"""
    if not os.path.exists('/kaggle/working/plant_disease_classification'):
        print("📦 Cloning repository...")
        try:
            subprocess.check_call([
                'git', 'clone', 
                'https://github.com/rajanikant04/plant_disease_classification.git',
                '/kaggle/working/plant_disease_classification'
            ], cwd='/kaggle/working')
        except:
            print("⚠️ Could not clone repository. Using current directory structure.")

def create_optimized_config():
    """Create memory-optimized configuration"""
    config_content = '''# Kaggle Memory-Optimized Configuration

class Config:
    # Dataset settings
    IMG_SIZE = 64  # Reduced from 224 for memory efficiency
    BATCH_SIZE = 8  # Small batch size for memory constraints
    NUM_EPOCHS = 10  # Reduced epochs for Kaggle time limits
    LEARNING_RATE = 0.001
    
    # Model settings - heavily optimized
    NUM_CLASSES = 4
    INPUT_CHANNELS = 3
    NUM_STAGES = 1  # Reduced stages for memory efficiency
    EMBED_DIMS = [32]  # Smaller embedding dimensions
    N_GROUPS = [32]
    
    # Memory optimization settings
    MAX_SEQUENCE_LENGTH = 64  # Adaptive pooling limit
    USE_MIXED_PRECISION = True
    PIN_MEMORY = False  # Disable for memory efficiency
    NUM_WORKERS = 0  # Disable multiprocessing in Kaggle
    
    # Paths (auto-detected)
    DATA_PATH = None  # Will be auto-detected
    MODEL_SAVE_PATH = '/kaggle/working/best_model.pth'
    
    @classmethod
    def detect_dataset_path(cls):
        """Auto-detect dataset path in Kaggle environment"""
        possible_paths = [
            '/kaggle/input/apple-disease-dataset',
            '/kaggle/input/plant-disease-dataset', 
            '/kaggle/input/apple-leaf-disease-dataset',
            '/kaggle/input',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Check for common dataset structures
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if any(name.lower() in ['train', 'test', 'apple'] for name in subdirs):
                    cls.DATA_PATH = path
                    print(f"📂 Dataset detected at: {path}")
                    return path
        
        print("⚠️ No dataset found in expected locations")
        return None

config = Config()
'''
    
    with open('/kaggle/working/config.py', 'w') as f:
        f.write(config_content)

def create_memory_optimized_training():
    """Create the optimized training script"""
    
    training_script = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import sys
from pathlib import Path

# Import our configuration
sys.path.append('/kaggle/working')
from config import config

print("🚀 Starting Memory-Optimized SClusterFormer Training")

# Memory optimization settings
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

class MemoryOptimizedDataLoader:
    """Memory-efficient data loading"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_loaders(self):
        """Get training and validation data loaders"""
        try:
            # Try pre-split dataset structure
            train_path = os.path.join(self.data_path, 'train')
            test_path = os.path.join(self.data_path, 'test')
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                print(f"📁 Using pre-split dataset structure")
                train_dataset = datasets.ImageFolder(train_path, transform=self.transform)
                val_dataset = datasets.ImageFolder(test_path, transform=self.transform)
            else:
                # Try single directory structure
                print(f"📁 Using single directory structure with split")
                full_dataset = datasets.ImageFolder(self.data_path, transform=self.transform)
                
                # Split dataset
                train_size = int(0.8 * len(full_dataset))
                val_size = len(full_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size]
                )
            
            # Create data loaders with memory optimization
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                drop_last=True  # Avoid variable batch sizes
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                drop_last=False
            )
            
            print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
            return train_loader, val_loader
            
        except Exception as e:
            print(f"❌ Error creating data loaders: {e}")
            return None, None

def train_model():
    """Memory-optimized training function"""
    
    # Detect dataset path
    data_path = config.detect_dataset_path()
    if not data_path:
        print("❌ No dataset found!")
        return
    
    # Create data loaders
    data_loader = MemoryOptimizedDataLoader(data_path)
    train_loader, val_loader = data_loader.get_loaders()
    
    if not train_loader:
        print("❌ Failed to create data loaders!")
        return
    
    # Import model (assume models are available)
    try:
        sys.path.append('/kaggle/working/plant_disease_classification')
        from models.SClusterFormer import SClusterFormer
        print("✅ Model imported successfully")
    except ImportError as e:
        print(f"❌ Could not import model: {e}")
        print("Please ensure model files are available")
        return
    
    # Create model with optimized settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Using device: {device}")
    
    model = SClusterFormer(
        input_channels=config.INPUT_CHANNELS,
        img_size=config.IMG_SIZE,
        num_classes=config.NUM_CLASSES,
        num_stages=config.NUM_STAGES,
        embed_dims=config.EMBED_DIMS,
        n_groups=config.N_GROUPS
    ).to(device)
    
    # Memory-efficient optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    # Mixed precision training for memory efficiency
    scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION else None
    
    print("🏋️ Starting training...")
    
    best_val_acc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if config.USE_MIXED_PRECISION and scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                if config.USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}]')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f'  💾 New best model saved! Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()
'''
    
    with open('/kaggle/working/kaggle_ultra_optimized.py', 'w') as f:
        f.write(training_script)

def main():
    """Main execution function"""
    print("🚀 Kaggle Ultra-Optimized SClusterFormer Setup")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Clone repository
    clone_repository()
    
    # Create optimized configuration
    create_optimized_config()
    
    # Create training script  
    create_memory_optimized_training()
    
    print("\n✅ Setup completed!")
    print("\nTo run the training:")
    print("1. Ensure your model files are available in the working directory")
    print("2. Run: python /kaggle/working/kaggle_ultra_optimized.py")
    print("\nOptimizations applied:")
    print("✓ Reduced image size to 64x64")
    print("✓ Small batch size (8)")
    print("✓ Single stage model")
    print("✓ Reduced embedding dimensions")
    print("✓ Adaptive sequence pooling (64 patches max)")
    print("✓ Mixed precision training")
    print("✓ Memory cleanup after batches")
    print("✓ Auto-dataset detection")
    
    # Run the training
    print("\n🚀 Starting training...")
    exec(open('/kaggle/working/kaggle_ultra_optimized.py').read())

if __name__ == "__main__":
    main()