import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as Data
from torchvision import transforms
from sklearn.model_selection import train_test_split


class PlantDiseaseDataset(torch.utils.data.Dataset):
    """Dataset class for RGB plant disease images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load RGB image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_plant_disease_data(data_dir, img_size=224, train_ratio=0.8, batch_size=32):
    """
    Load plant disease dataset from directory structure:
    data_dir/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg
            img4.jpg
    """
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    class_names = []
    
    if os.path.exists(data_dir):
        for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_names.append(class_name)
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_file)
                        image_paths.append(img_path)
                        labels.append(class_idx)
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    print(f"Found {len(image_paths)} images in {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, 
        train_size=train_ratio, 
        stratify=labels,
        random_state=42
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(X_train, y_train, transform=train_transform)
    test_dataset = PlantDiseaseDataset(X_test, y_test, transform=test_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    return train_loader, test_loader, len(class_names), class_names


def load_kaggle_plant_disease_data(kaggle_input_path="/kaggle/input/apple-disease-dataset", 
                                 img_size=224, train_ratio=0.8, batch_size=32, 
                                 use_validation_split=True):
    """
    Load plant disease data from Kaggle input directory
    
    Args:
        kaggle_input_path: Path to Kaggle dataset
        img_size: Image size for resizing
        train_ratio: Ratio for train/val split (only used if splitting train folder)
        batch_size: Batch size for data loaders
        use_validation_split: If True, splits train folder into train/val
    """
    datasets_dir = os.path.join(kaggle_input_path, "datasets")
    
    if not os.path.exists(datasets_dir):
        # Try direct path without 'datasets' folder
        datasets_dir = kaggle_input_path
        
    if not os.path.exists(datasets_dir):
        print(f"Dataset directory not found: {datasets_dir}")
        return None, None, 0, []
    
    # Check for train/test split structure
    train_dir = os.path.join(datasets_dir, "train")
    test_dir = os.path.join(datasets_dir, "test")
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"✅ Found pre-split train/test structure")
        print(f"📁 Train directory: {train_dir}")
        print(f"📁 Test directory: {test_dir}")
        
        if use_validation_split:
            # Split train directory into train/val (80%/20%)
            print(f"🔄 Splitting train folder: {train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% validation")
            train_loader, val_loader = create_train_val_loaders_from_dir(
                train_dir, img_size, batch_size, train_ratio
            )
            test_loader = create_loader_from_dir(test_dir, img_size, batch_size, is_train=False)
            
            print(f"📊 Final split: Train + Val (from train folder), Test (from test folder)")
            
        else:
            # Use original train/test without validation split
            print(f"📊 Using original train/test split without validation")
            train_loader = create_loader_from_dir(train_dir, img_size, batch_size, is_train=True)
            val_loader = create_loader_from_dir(test_dir, img_size, batch_size, is_train=False)
            test_loader = val_loader  # Same as validation
        
        # Count classes from train directory
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        num_classes = len(class_names)
        
        print(f"🏷️  Found {num_classes} classes: {class_names}")
        
        return train_loader, val_loader, num_classes, class_names
    
    else:
        # Single directory, need to split
        print("🔄 Single directory found - splitting into train/test")
        return load_plant_disease_data(datasets_dir, img_size, train_ratio, batch_size)


def create_loader_from_dir(data_dir, img_size, batch_size, is_train=True):
    """Create data loader from directory structure"""
    
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_file)
                    image_paths.append(img_path)
                    labels.append(class_idx)
    
    # Create dataset
    dataset = PlantDiseaseDataset(image_paths, labels, transform=transform)
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        drop_last=is_train
    )
    
    print(f"Created {'train' if is_train else 'test'} loader with {len(dataset)} samples")
    return loader


def create_train_val_loaders_from_dir(data_dir, img_size, batch_size, train_ratio=0.8):
    """
    Create train and validation loaders by splitting the train directory
    
    Args:
        data_dir: Directory containing class subdirectories
        img_size: Image size for resizing
        batch_size: Batch size for data loaders
        train_ratio: Fraction of data to use for training (rest for validation)
    
    Returns:
        train_loader, val_loader
    """
    
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect all image paths and labels
    all_image_paths = []
    all_labels = []
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        class_images = []
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_file)
                class_images.append(img_path)
        
        all_image_paths.extend(class_images)
        all_labels.extend([class_idx] * len(class_images))
        
        print(f"  📂 {class_name}: {len(class_images)} images")
    
    # Convert to numpy for easier manipulation
    all_image_paths = np.array(all_image_paths)
    all_labels = np.array(all_labels)
    
    # Split into train and validation sets (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        all_image_paths, all_labels,
        train_size=train_ratio,
        stratify=all_labels,
        random_state=42
    )
    
    print(f"  📊 Train: {len(X_train)} samples ({train_ratio*100:.1f}%)")
    print(f"  📊 Validation: {len(X_val)} samples ({(1-train_ratio)*100:.1f}%)")
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(X_train, y_train, transform=train_transform)
    val_dataset = PlantDiseaseDataset(X_val, y_val, transform=val_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing RGB data loader...")
    
    # Example usage
    data_dir = "path/to/your/plant/disease/dataset"
    train_loader, test_loader, num_classes, class_names = load_plant_disease_data(
        data_dir=data_dir,
        img_size=224,
        train_ratio=0.8,
        batch_size=16
    )
    
    if train_loader is not None:
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
        
        # Test loading a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
            if batch_idx >= 2:  # Only test a few batches
                break