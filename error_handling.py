"""
🛡️ Comprehensive Error Handling for Kaggle Environment
Robust error recovery and fallback mechanisms
"""

import os
import sys
import traceback
import warnings
import time
from contextlib import contextmanager

class KaggleErrorHandler:
    """
    Centralized error handling for Kaggle deployment
    """
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.fallbacks_used = []
    
    def log_error(self, error_type, message, suggestion=None):
        """Log an error with optional suggestion"""
        error_info = {
            'type': error_type,
            'message': str(message),
            'suggestion': suggestion,
            'timestamp': time.time()
        }
        self.errors.append(error_info)
        
        print(f"❌ {error_type}: {message}")
        if suggestion:
            print(f"💡 Suggestion: {suggestion}")
    
    def log_warning(self, warning_type, message):
        """Log a warning"""
        warning_info = {
            'type': warning_type,
            'message': str(message),
            'timestamp': time.time()
        }
        self.warnings.append(warning_info)
        print(f"⚠️ {warning_type}: {message}")
    
    def log_fallback(self, fallback_type, reason):
        """Log when a fallback is used"""
        fallback_info = {
            'type': fallback_type,
            'reason': reason,
            'timestamp': time.time()
        }
        self.fallbacks_used.append(fallback_info)
        print(f"🔄 Fallback: {fallback_type} - {reason}")
    
    @contextmanager
    def error_context(self, operation_name):
        """Context manager for error handling"""
        try:
            print(f"🚀 Starting: {operation_name}")
            yield
            print(f"✅ Completed: {operation_name}")
        except Exception as e:
            self.log_error(
                error_type=f"{operation_name} Error",
                message=str(e),
                suggestion=f"Check {operation_name.lower()} configuration and try again"
            )
            raise
    
    def get_summary(self):
        """Get error summary"""
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'fallbacks_used': len(self.fallbacks_used),
            'errors': self.errors,
            'warnings': self.warnings,
            'fallbacks': self.fallbacks_used
        }

# Global error handler instance
error_handler = KaggleErrorHandler()

def safe_import(module_name, fallback_name=None, error_handler=None):
    """
    Safely import a module with optional fallback
    """
    if error_handler is None:
        error_handler = globals().get('error_handler', KaggleErrorHandler())
    
    try:
        module = __import__(module_name, fromlist=[''])
        print(f"✅ Successfully imported {module_name}")
        return module
    except ImportError as e:
        error_handler.log_error(
            error_type="Import Error",
            message=f"Could not import {module_name}: {e}",
            suggestion=f"Check if {module_name} is available in Kaggle environment"
        )
        
        if fallback_name:
            try:
                fallback_module = __import__(fallback_name, fromlist=[''])
                error_handler.log_fallback(
                    fallback_type="Module Import",
                    reason=f"Using {fallback_name} instead of {module_name}"
                )
                return fallback_module
            except ImportError as e2:
                error_handler.log_error(
                    error_type="Fallback Import Error",
                    message=f"Could not import fallback {fallback_name}: {e2}",
                    suggestion="Consider using emergency fallback implementation"
                )
        
        return None

def safe_torch_operations():
    """
    Safely configure PyTorch with error handling
    """
    try:
        import torch
        
        # Memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set conservative memory settings
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
            error_handler.log_warning("CUDA Settings", "Applied conservative memory settings")
        
        # Disable benchmarking for stability
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Suppress warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        print("✅ PyTorch configured safely")
        return True
        
    except Exception as e:
        error_handler.log_error(
            error_type="PyTorch Configuration Error",
            message=str(e),
            suggestion="Try running without CUDA or with smaller batch sizes"
        )
        return False

def safe_dataset_detection(base_paths=None):
    """
    Safely detect dataset with comprehensive error handling
    """
    if base_paths is None:
        base_paths = [
            "/kaggle/input",
            "/content/drive/MyDrive",  # Colab fallback
            "./data",  # Local fallback
            "."  # Current directory
        ]
    
    dataset_path = None
    
    for base_path in base_paths:
        try:
            if not os.path.exists(base_path):
                continue
                
            print(f"🔍 Checking {base_path}")
            items = os.listdir(base_path)
            
            for item in items:
                item_path = os.path.join(base_path, item)
                
                if not os.path.isdir(item_path):
                    continue
                
                # Check for dataset indicators
                if any(keyword in item.lower() for keyword in ['plant', 'disease', 'apple', 'leaf']):
                    # Verify dataset structure
                    subdirs = os.listdir(item_path)
                    
                    # Check for datasets folder
                    if 'datasets' in subdirs:
                        dataset_path = os.path.join(item_path, 'datasets')
                        print(f"✅ Found dataset with 'datasets' folder: {dataset_path}")
                        return dataset_path
                    
                    # Check for train/test structure
                    if any(d in subdirs for d in ['train', 'test']):
                        dataset_path = item_path
                        print(f"✅ Found dataset with train/test structure: {dataset_path}")
                        return dataset_path
                    
                    # Check for class folders (at least 2 directories)
                    class_dirs = [d for d in subdirs if os.path.isdir(os.path.join(item_path, d))]
                    if len(class_dirs) >= 2:
                        dataset_path = item_path
                        print(f"✅ Found dataset with class folders: {dataset_path}")
                        return dataset_path
            
        except PermissionError:
            error_handler.log_warning("Permission Warning", f"Cannot access {base_path}")
        except Exception as e:
            error_handler.log_error(
                error_type="Dataset Detection Error",
                message=f"Error checking {base_path}: {e}",
                suggestion="Verify dataset path and permissions"
            )
    
    error_handler.log_error(
        error_type="Dataset Not Found",
        message="No suitable dataset found in any expected location",
        suggestion="Check dataset upload and folder structure"
    )
    
    return None

def create_emergency_dataset(output_path="./emergency_data"):
    """
    Create a minimal dataset for testing when no real dataset is found
    """
    try:
        import torch
        from PIL import Image
        import numpy as np
        
        print(f"🔧 Creating emergency dataset at {output_path}")
        
        # Create directory structure
        classes = ['healthy', 'diseased', 'rust', 'scab']
        
        for split in ['train', 'test']:
            for class_name in classes:
                class_dir = os.path.join(output_path, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Create 5 synthetic images per class
                for i in range(5):
                    # Create random RGB image
                    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    
                    img_path = os.path.join(class_dir, f"synthetic_{i}.jpg")
                    img.save(img_path)
        
        print(f"✅ Emergency dataset created with {len(classes)} classes")
        error_handler.log_fallback(
            fallback_type="Emergency Dataset",
            reason="No real dataset found, created synthetic data for testing"
        )
        
        return output_path
        
    except Exception as e:
        error_handler.log_error(
            error_type="Emergency Dataset Error",
            message=f"Could not create emergency dataset: {e}",
            suggestion="Check write permissions and available space"
        )
        return None

def memory_check_and_optimize():
    """
    Check available memory and apply optimizations
    """
    try:
        import torch
        import psutil
        
        # Check system memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        print(f"💾 Available RAM: {available_gb:.1f} GB")
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 8:  # Less than 8GB GPU
                error_handler.log_warning(
                    "Low GPU Memory",
                    f"GPU has only {gpu_memory:.1f}GB, applying conservative settings"
                )
                return {
                    'batch_size': 2,
                    'img_size': 64,
                    'use_mixed_precision': False,
                    'num_workers': 0
                }
        
        if available_gb < 4:  # Less than 4GB RAM
            error_handler.log_warning(
                "Low System Memory",
                f"System has only {available_gb:.1f}GB RAM, applying minimal settings"
            )
            return {
                'batch_size': 1,
                'img_size': 32,
                'use_mixed_precision': False,
                'num_workers': 0
            }
        
        # Default optimized settings
        return {
            'batch_size': 4,
            'img_size': 64,
            'use_mixed_precision': True,
            'num_workers': 0
        }
        
    except Exception as e:
        error_handler.log_error(
            error_type="Memory Check Error",
            message=str(e),
            suggestion="Proceeding with minimal settings"
        )
        
        return {
            'batch_size': 1,
            'img_size': 32,
            'use_mixed_precision': False,
            'num_workers': 0
        }

def print_error_summary():
    """
    Print a summary of all errors and fallbacks used
    """
    summary = error_handler.get_summary()
    
    print("\n" + "="*60)
    print("🛡️ ERROR HANDLING SUMMARY")
    print("="*60)
    
    print(f"📊 Total Errors: {summary['total_errors']}")
    print(f"⚠️ Total Warnings: {summary['total_warnings']}")
    print(f"🔄 Fallbacks Used: {summary['fallbacks_used']}")
    
    if summary['errors']:
        print("\n❌ ERRORS:")
        for i, error in enumerate(summary['errors'], 1):
            print(f"  {i}. {error['type']}: {error['message']}")
            if error['suggestion']:
                print(f"     💡 {error['suggestion']}")
    
    if summary['fallbacks']:
        print("\n🔄 FALLBACKS USED:")
        for i, fallback in enumerate(summary['fallbacks'], 1):
            print(f"  {i}. {fallback['type']}: {fallback['reason']}")
    
    if summary['warnings']:
        print("\n⚠️ WARNINGS:")
        for i, warning in enumerate(summary['warnings'], 1):
            print(f"  {i}. {warning['type']}: {warning['message']}")
    
    print("="*60)

if __name__ == "__main__":
    # Test error handling functionality
    print("🧪 Testing error handling...")
    
    # Test safe import
    torch_module = safe_import('torch')
    fake_module = safe_import('nonexistent_module', 'torch')
    
    # Test dataset detection
    dataset_path = safe_dataset_detection()
    
    # Test memory optimization
    memory_settings = memory_check_and_optimize()
    print(f"💾 Memory settings: {memory_settings}")
    
    # Print summary
    print_error_summary()
    
    print("\n✅ Error handling tests completed!")