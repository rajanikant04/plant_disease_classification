"""
RGB Plant Disease Classification Training Loop
Simplified version of Loop_EMAP_train.py adapted for RGB images
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

from models.SClusterFormer import SClusterFormer
from rgb_data_loader import load_kaggle_plant_disease_data, load_plant_disease_data


def train_rgb_sclusterformer(hyperparameters):
    """
    Train SClusterFormer on RGB plant disease images
    
    Args:
        hyperparameters: dict containing training parameters
    """
    
    # Extract hyperparameters
    img_size = hyperparameters.get('img_size', 224)
    input_channels = hyperparameters.get('input_channels', 3)
    batch_size = hyperparameters.get('batch_size', 32)
    epochs = hyperparameters.get('epochs', 100)
    learning_rate = hyperparameters.get('learning_rate', 0.001)
    device_ids = hyperparameters.get('cuda', [0])
    data_path = hyperparameters.get('data_path', '/kaggle/input/apple-disease-dataset')
    train_ratio = hyperparameters.get('train_ratio', 0.8)
    use_validation_split = hyperparameters.get('use_validation_split', True)
    model_config = hyperparameters.get('model_config', {
        'num_stages': 3,
        'n_groups': [16, 16, 16],
        'embed_dims': [256, 128, 64],
        'num_heads': [8, 4, 2],
        'mlp_ratios': [4, 4, 4],
        'depths': [2, 2, 2],
    })
    
    # Set device
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading RGB plant disease dataset...")
    
    if os.path.exists(data_path):
        if 'kaggle' in data_path:
            train_loader, test_loader, num_classes, class_names = load_kaggle_plant_disease_data(
                kaggle_input_path=data_path,
                img_size=img_size,
                train_ratio=train_ratio,
                batch_size=batch_size,
                use_validation_split=use_validation_split
            )
        else:
            train_loader, test_loader, num_classes, class_names = load_plant_disease_data(
                data_dir=data_path,
                img_size=img_size,
                train_ratio=train_ratio,
                batch_size=batch_size
            )
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    if train_loader is None:
        raise ValueError("Failed to create data loaders")
    
    print(f"Dataset loaded: {num_classes} classes")
    print(f"Classes: {class_names}")
    
    # Initialize model
    model = SClusterFormer(
        img_size=img_size,
        input_channels=input_channels,
        num_classes=num_classes,
        num_stages=model_config['num_stages'],
        n_groups=model_config['n_groups'],
        embed_dims=model_config['embed_dims'],
        num_heads=model_config['num_heads'],
        mlp_ratios=model_config['mlp_ratios'],
        depths=model_config['depths'],
        patchsize=img_size//8
    )
    
    # Move to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Memory optimization
    if device.type == 'cpu' or not torch.cuda.is_available():
        print("🔧 CPU mode: Using memory optimizations")
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    # Training
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Memory cleanup at start of epoch
        if epoch % 2 == 0:  # Every 2 epochs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import gc
            gc.collect()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Delete intermediate tensors to save memory
            del outputs, loss
            
            # Periodic memory cleanup during training
            if batch_idx % 10 == 0 and batch_idx > 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        test_accuracies.append(test_acc)
        
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}] '
                  f'Loss: {epoch_loss:.4f} '
                  f'Train Acc: {epoch_acc:.2f}% '
                  f'Test Acc: {test_acc:.2f}%')
        
        # Early stopping
        if epoch_loss < 0.01:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("\nFinal evaluation...")
    test_start_time = time.time()
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    testing_time = time.time() - test_start_time
    
    # Calculate metrics
    oa = accuracy_score(all_labels, all_predictions)
    confusion = confusion_matrix(all_labels, all_predictions)
    each_acc, aa = calculate_class_accuracies(confusion)
    kappa = cohen_kappa_score(all_labels, all_predictions)
    classification_rep = classification_report(all_labels, all_predictions, 
                                             target_names=class_names, digits=4)
    
    # Print results
    print(f"\n{'='*60}")
    print("TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Testing Time: {testing_time:.2f}s")
    print(f"Overall Accuracy (OA): {oa:.4f}")
    print(f"Average Accuracy (AA): {aa:.4f}")
    print(f"Kappa Score: {kappa:.4f}")
    print("\nPer-class Accuracies:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {each_acc[i]:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_rep)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(train_losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Training accuracy
    axes[0, 1].plot(train_accuracies, label='Train Acc')
    axes[0, 1].plot(test_accuracies, label='Test Acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Confusion matrix
    im = axes[1, 0].imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 0])
    
    # Per-class accuracy bar chart
    axes[1, 1].bar(range(len(class_names)), each_acc)
    axes[1, 1].set_title('Per-class Accuracy')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_xticks(range(len(class_names)))
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), 'sclusterformer_rgb_model.pth')
    print(f"\nModel saved as 'sclusterformer_rgb_model.pth'")
    
    # Save results to file
    with open('results.txt', 'w') as f:
        f.write("SClusterFormer RGB Plant Disease Classification Results\n")
        f.write("="*60 + "\n")
        f.write(f"Overall Accuracy: {oa:.4f}\n")
        f.write(f"Average Accuracy: {aa:.4f}\n")
        f.write(f"Kappa Score: {kappa:.4f}\n")
        f.write(f"Training Time: {training_time:.2f}s\n")
        f.write(f"Testing Time: {testing_time:.2f}s\n")
        f.write("\nPer-class Accuracies:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}: {each_acc[i]:.4f}\n")
        f.write(f"\nClassification Report:\n{classification_rep}\n")
    
    return {
        'oa': oa,
        'aa': aa,
        'kappa': kappa,
        'each_acc': each_acc,
        'training_time': training_time,
        'testing_time': testing_time,
        'model': model,
        'confusion_matrix': confusion,
        'class_names': class_names
    }


def calculate_class_accuracies(confusion_matrix):
    """Calculate per-class accuracies from confusion matrix"""
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def loop_train_test_rgb(hyperparameters):
    """
    Run multiple training experiments with RGB data
    Compatible with original loop_train_test function signature
    """
    
    run_times = hyperparameters.get('run_times', 1)
    
    results = {
        'oa_list': [],
        'aa_list': [],
        'kappa_list': [],
        'class_acc_list': [],
        'train_times': [],
        'test_times': []
    }
    
    print(f"Running {run_times} experiments...")
    
    for run_idx in range(run_times):
        print(f"\n{'='*50}")
        print(f"RUN {run_idx + 1}/{run_times}")
        print(f"{'='*50}")
        
        result = train_rgb_sclusterformer(hyperparameters)
        
        results['oa_list'].append(result['oa'])
        results['aa_list'].append(result['aa'])
        results['kappa_list'].append(result['kappa'])
        results['class_acc_list'].append(result['each_acc'])
        results['train_times'].append(result['training_time'])
        results['test_times'].append(result['testing_time'])
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL RUNS")
    print(f"{'='*70}")
    
    oa_array = np.array(results['oa_list'])
    aa_array = np.array(results['aa_list'])
    kappa_array = np.array(results['kappa_list'])
    class_acc_array = np.array(results['class_acc_list'])
    
    print(f"Overall Accuracy: {np.mean(oa_array):.4f} ± {np.std(oa_array):.4f}")
    print(f"Average Accuracy: {np.mean(aa_array):.4f} ± {np.std(aa_array):.4f}")
    print(f"Kappa Score: {np.mean(kappa_array):.4f} ± {np.std(kappa_array):.4f}")
    print(f"Training Time: {np.mean(results['train_times']):.2f} ± {np.std(results['train_times']):.2f}s")
    print(f"Testing Time: {np.mean(results['test_times']):.2f} ± {np.std(results['test_times']):.2f}s")
    
    return results


# Compatibility function for original interface
def loop_train_test(hyperparameters):
    """
    Wrapper function to maintain compatibility with original main.py
    """
    # Convert old hyperparameters to new format
    new_hp = {
        'img_size': 224,
        'input_channels': 3,
        'batch_size': hyperparameters.get('BATCH_SIZE_TRAIN', 32),
        'epochs': hyperparameters.get('epochs', 100),
        'learning_rate': 0.001,
        'cuda': hyperparameters.get('cuda', [0]),
        'run_times': hyperparameters.get('run_times', 1),
        'train_ratio': hyperparameters.get('train_ratio', 0.8),
        'data_path': '/kaggle/input/apple-disease-dataset'  # Default Kaggle path
    }
    
    print("Converting hyperspectral parameters to RGB parameters...")
    print("Note: This is now adapted for RGB plant disease classification")
    
    return loop_train_test_rgb(new_hp)


if __name__ == "__main__":
    # Test the training loop
    test_hp = {
        'img_size': 224,
        'input_channels': 3,
        'batch_size': 16,
        'epochs': 5,  # Small for testing
        'learning_rate': 0.001,
        'cuda': [0],
        'run_times': 1,
        'train_ratio': 0.8,
        'data_path': '/kaggle/input/apple-disease-dataset'
    }
    
    result = train_rgb_sclusterformer(test_hp)
    print("Test completed successfully!")