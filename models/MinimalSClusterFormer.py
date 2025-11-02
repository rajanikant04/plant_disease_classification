"""
🔧 Minimal SClusterFormer Fallback Model
A simple CNN-based replacement when the full SClusterFormer model fails to load
Designed for maximum Kaggle compatibility with minimal dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalSClusterFormer(nn.Module):
    """
    Ultra-minimal CNN-based fallback model that mimics SClusterFormer interface
    Uses only basic PyTorch operations for maximum compatibility
    """
    
    def __init__(self, input_channels=3, img_size=64, num_classes=4, **kwargs):
        super().__init__()
        
        self.input_channels = input_channels
        self.img_size = img_size
        self.num_classes = num_classes
        
        print(f"🔧 MinimalSClusterFormer: {input_channels}→{num_classes}, size={img_size}")
        
        # Simple feature extraction backbone
        self.feature_extractor = nn.Sequential(
            # First conv block - 64→32
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Second conv block - 32→16  
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Third conv block - 16→8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Fourth conv block - 8→4
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simple classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Input: [B, 3, H, W]
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.feature_extractor(x)  # [B, 64, 1, 1]
        
        # Flatten
        features = features.view(batch_size, -1)  # [B, 64]
        
        # Classification
        output = self.classifier(features)  # [B, num_classes]
        
        return output
    
    def get_model_info(self):
        """Get model information for debugging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MinimalSClusterFormer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'img_size': self.img_size
        }

class AdaptiveMinimalModel(nn.Module):
    """
    Even more minimal model for extreme memory constraints
    """
    
    def __init__(self, input_channels=3, num_classes=4, **kwargs):
        super().__init__()
        
        print("🔧 AdaptiveMinimalModel: Ultra-light CNN")
        
        # Ultra-minimal architecture
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 16, 7, stride=4, padding=3),  # 64→16
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),  # 16→8
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((2, 2)),  # 8→2
            nn.Flatten(),
            nn.Linear(32 * 4, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_fallback_model(input_channels=3, img_size=64, num_classes=4, **kwargs):
    """
    Factory function to create the appropriate fallback model
    """
    
    # Estimate memory requirements
    estimated_memory = (img_size ** 2) * input_channels * 4 / (1024 ** 2)  # MB per image
    
    print(f"🔍 Estimated memory per image: {estimated_memory:.2f} MB")
    
    if estimated_memory > 50:  # Very large images
        print("⚠️ Large image detected - using AdaptiveMinimalModel")
        return AdaptiveMinimalModel(input_channels=input_channels, num_classes=num_classes)
    else:
        print("✅ Using MinimalSClusterFormer")
        return MinimalSClusterFormer(
            input_channels=input_channels,
            img_size=img_size,
            num_classes=num_classes,
            **kwargs
        )

# Compatibility aliases for import
SClusterFormer = MinimalSClusterFormer

if __name__ == "__main__":
    # Test the minimal model
    print("🧪 Testing MinimalSClusterFormer...")
    
    # Test with different configurations
    configs = [
        {"input_channels": 3, "img_size": 64, "num_classes": 4},
        {"input_channels": 3, "img_size": 96, "num_classes": 4},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n📋 Test {i}: {config}")
        
        model = create_fallback_model(**config)
        
        # Test input
        batch_size = 2
        x = torch.randn(batch_size, config["input_channels"], config["img_size"], config["img_size"])
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
            print(f"  Input: {list(x.shape)}")
            print(f"  Output: {list(output.shape)}")
            
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                print(f"  Parameters: {info['total_parameters']:,}")
        
        print("  ✅ Success!")
    
    print("\n🎉 All fallback models working correctly!")