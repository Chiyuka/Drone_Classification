import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import numpy as np
import os

# --- Configuration ---
FINAL_DATASET_PATH = 'drone_dataset_spectrograms/final_labeled_dataset.csv'
BATCH_SIZE = 64
NUM_CLASSES = 3  # 0: Mavic_1, 1: Mavic_2, 2: Mavic_Mini
LEARNING_RATE = 0.0001
NUM_EPOCHS = 15
PATIENCE = 5

# ImageNet normalization for ResNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -------------------------------------------------------------
# Part A: Custom PyTorch Dataset (Loads the.pt files)
# -------------------------------------------------------------
class DroneAcousticDataset(Dataset):
    def __init__(self, metadata_df, normalize=True):
        self.metadata = metadata_df
        self.normalize = normalize
        
        # Mapping for 3 drone types only
        self.label_map = {
            'Mavic_1': 0,
            'Mavic_2': 1, 
            'Mavic_Mini': 2
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. Load the Spectrogram Tensor
        feature_path = row['feature_path']
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
            
        spectrogram = torch.load(feature_path).float()
        
        if idx == 0:
            print(f"DEBUG - Raw spectrogram: shape={spectrogram.shape}, range=[{spectrogram.min():.3f}, {spectrogram.max():.3f}]")
            print(f"DEBUG - Raw mean={spectrogram.mean():.3f}, std={spectrogram.std():.3f}")
        
        # 2. Handle different tensor dimensions
        if spectrogram.dim() == 2:
            # 2D tensor: [n_mels, time] -> [1, n_mels, time]
            spectrogram = spectrogram.unsqueeze(0)
        
        # 3. Apply normalization (more robust)
        if self.normalize:
            # Simple min-max normalization to [0, 1]
            eps = 1e-8  # Small epsilon to avoid division by zero
            spectrogram_min = spectrogram.min()
            spectrogram_max = spectrogram.max()
            
            # Check if tensor has variation
            if spectrogram_max - spectrogram_min > eps:
                spectrogram = (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min + eps)
            else:
                # If all values are the same, set to 0.5
                spectrogram = torch.zeros_like(spectrogram) + 0.5
            
            # Apply ImageNet normalization for ResNet
            for i in range(min(spectrogram.shape[0], 3)):  # Handle up to 3 channels
                spectrogram[i] = (spectrogram[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
        
        if idx == 0:
            print(f"DEBUG - Normalized spectrogram: shape={spectrogram.shape}, range=[{spectrogram.min():.3f}, {spectrogram.max():.3f}]")
        
        # 4. Convert to 3 channels for ResNet
        if spectrogram.shape[0] == 1:
            # Single channel -> repeat to 3 channels
            spectrogram_3ch = spectrogram.repeat(3, 1, 1)
        elif spectrogram.shape[0] == 3:
            spectrogram_3ch = spectrogram
        else:
            # If more than 3 channels, take first 3
            if spectrogram.shape[0] > 3:
                spectrogram_3ch = spectrogram[:3, :, :]
                print(f"Warning: Cropping channels from {spectrogram.shape[0]} to 3")
            else:
                # If 2 channels, add a third zero channel
                zeros = torch.zeros_like(spectrogram[0:1, :, :])
                spectrogram_3ch = torch.cat([spectrogram, zeros], dim=0)
        
        # 5. Extract and Prepare Label
        label_string = str(row['label_type']).strip()
        
        if label_string in self.label_map:
            label_idx = self.label_map[label_string]
        else:
            print(f"Warning: Unexpected label '{label_string}' at index {idx}, using Mavic_1")
            label_idx = 0
        
        label = torch.tensor(label_idx, dtype=torch.long)
        
        return spectrogram_3ch, label

# -------------------------------------------------------------
# Part B: Model Setup
# -------------------------------------------------------------
def setup_model(device, class_counts=None):
    # Create the model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.float()
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layers from layer3 onwards
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name:
            param.requires_grad = True
    
    # Modify the final layer for 3 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, NUM_CLASSES)
    )
    
    # Initialize weights properly
    for layer in model.fc:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {trainable_params:,} trainable parameters out of {total_params:,} total")
    
    return model.to(device)

# -------------------------------------------------------------
# Part C: Training Function
# -------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, class_counts):
    # Calculate class weights
    num_classes = len(class_counts)
    weights = []
    for count in class_counts:
        if count == 0:
            weight = 0.0
        else:
            weight = 1.0 / count
        weights.append(weight)
    
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes
    
    class_names = ['Mavic_1', 'Mavic_2', 'Mavic_Mini']
    print(f"\nClass weights:")
    for i in range(num_classes):
        print(f"  {class_names[i]}: count={class_counts[i]}, weight={weights[i]:.3f}")
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Pre-training diagnostics
    print(f"\n=== PRE-TRAINING DIAGNOSTICS ===")
    model.eval()
    with torch.no_grad():
        sample_batch, sample_labels = next(iter(train_loader))
        sample_batch, sample_labels = sample_batch.to(device), sample_labels.to(device)
        
        print(f"\n1. Input batch shape: {sample_batch.shape}")
        print(f"   Input range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
        print(f"   Input mean: {sample_batch.mean():.3f}, std: {sample_batch.std():.3f}")
        
        outputs = model(sample_batch)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
        
        print(f"\n2. Model outputs shape: {outputs.shape}")
        print(f"   Outputs range: [{outputs.min():.3f}, {outputs.max():.3f}]")
        
        print(f"\n3. Predictions vs Actual labels:")
        for i in range(num_classes):
            pred_count = (predictions == i).sum().item()
            actual_count = (sample_labels == i).sum().item()
            print(f"   {class_names[i]}: Predicted {pred_count}, Actual {actual_count}")
    
    model.train()
    
    # Training loop
    best_balanced_acc = 0
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_loop.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_accuracy = 100 * correct_train / total_train
        avg_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                for i in range(num_classes):
                    class_mask = (labels == i)
                    class_total[i] += class_mask.sum().item()
                    class_correct[i] += (predicted[class_mask] == labels[class_mask]).sum().item()
        
        val_accuracy = 100 * correct_val / total_val
        
        # Calculate per-class and balanced accuracy
        class_accuracies = []
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0.0)
        
        balanced_acc = sum(class_accuracies) / num_classes
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
        print(f"  Balanced Accuracy: {balanced_acc:.2f}%")
        print(f"  Per-class accuracy:")
        for i in range(num_classes):
            print(f"    {class_names[i]}: {class_accuracies[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_drone_detector.pth')
            print(f"  ✓ New best model saved! (Balanced Acc: {balanced_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Best Balanced accuracy: {best_balanced_acc:.2f}%")
    
    # Load best model
    if os.path.exists('best_drone_detector.pth'):
        model.load_state_dict(torch.load('best_drone_detector.pth'))
        print("Loaded best model weights.")
    
    return model

# -------------------------------------------------------------
# Part D: Custom Collate Function (for variable time lengths)
# -------------------------------------------------------------
def custom_collate_fn(batch):
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    
    spectrograms = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Find max time dimension
    max_time = max(spec.shape[2] for spec in spectrograms)
    
    # Pad if needed
    padded_spectrograms = []
    for spec in spectrograms:
        if spec.shape[2] < max_time:
            padding_needed = max_time - spec.shape[2]
            padded_spec = torch.nn.functional.pad(spec, (0, padding_needed))
            padded_spectrograms.append(padded_spec)
        else:
            padded_spectrograms.append(spec)
    
    return torch.stack(padded_spectrograms), torch.stack(labels)

# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
if __name__ == '__main__':
    # Set random seeds
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_seed(42)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load data
    final_df = pd.read_csv(FINAL_DATASET_PATH)
    
    print(f"\n=== CHECKING FOR NAN VALUES ===")
    nan_count = final_df['label_type'].isna().sum()
    print(f"NaN values in label_type: {nan_count}")
    
    if nan_count > 0:
        print(f"Removing {nan_count} rows with NaN labels...")
        final_df = final_df.dropna(subset=['label_type'])
        print(f"Remaining samples: {len(final_df)}")
    
    print(f"\n=== DATASET ANALYSIS ===")
    print(f"Total samples: {len(final_df)}")
    
    # Show distribution
    label_counts = final_df['label_type'].value_counts()
    for label, count in label_counts.items():
        print(f"{label}: {count} ({100*count/len(final_df):.1f}%)")
    
    # 2. DEBUG: Check spectrogram shapes
    print(f"\n=== DEBUGGING SPECTROGRAM SHAPES ===")
    shape_counts = {}
    for i in range(min(5, len(final_df))):
        try:
            file_path = final_df.iloc[i]['feature_path']
            spec = torch.load(file_path)
            shape_str = str(tuple(spec.shape))
            shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1
            print(f"File {i}: {file_path}")
            print(f"  Shape: {spec.shape}, Dim: {spec.dim()}")
            print(f"  Range: [{spec.min():.3f}, {spec.max():.3f}]")
            print("-" * 40)
        except Exception as e:
            print(f"Error loading file {i}: {e}")
    
    # Check random samples too
    print(f"\n=== RANDOM SAMPLE CHECK ===")
    import random
    random_indices = random.sample(range(len(final_df)), 3)
    for idx in random_indices:
        try:
            file_path = final_df.iloc[idx]['feature_path']
            spec = torch.load(file_path)
            print(f"Random file {idx}: shape={spec.shape}, dim={spec.dim()}")
        except Exception as e:
            print(f"Error loading random file {idx}: {e}")
    
    # 3. Check file quality
    print(f"\n=== CHECKING FILE QUALITY ===")
    bad_files = []
    good_files = 0
    
    # First, let's find what shapes we have
    shape_counts = {}
    for idx, row in final_df.iterrows():
        try:
            spectrogram = torch.load(row['feature_path'])
            shape_str = str(tuple(spectrogram.shape))
            shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1
        except:
            pass
    
    print("Spectrogram shapes found:")
    for shape, count in sorted(shape_counts.items()):
        print(f"  Shape {shape}: {count} files")
    
    # Now check quality with more flexible criteria
    print(f"\n=== QUALITY CHECK ===")
    for idx, row in final_df.iterrows():
        try:
            spectrogram = torch.load(row['feature_path'])
            
            # Accept if tensor has reasonable dimensions
            if spectrogram.dim() == 2:
                # 2D tensor: [n_mels, time]
                n_mels, time_steps = spectrogram.shape
                # More flexible: accept if has at least some content
                if time_steps >= 10 and n_mels >= 10:  # Reduced from 50
                    good_files += 1
                else:
                    bad_files.append(row['feature_path'])
                    print(f"Rejected (too small): {row['feature_path']} - shape {spectrogram.shape}")
            elif spectrogram.dim() == 3:
                # 3D tensor: [channels, n_mels, time]
                channels, n_mels, time_steps = spectrogram.shape
                if time_steps >= 10 and n_mels >= 10:
                    good_files += 1
                else:
                    bad_files.append(row['feature_path'])
                    print(f"Rejected (too small): {row['feature_path']} - shape {spectrogram.shape}")
            else:
                bad_files.append(row['feature_path'])
                print(f"Rejected (wrong dims): {row['feature_path']} - shape {spectrogram.shape}")
        except Exception as e:
            print(f"Error loading {row['feature_path']}: {e}")
            bad_files.append(row['feature_path'])
    
    print(f"Good files: {good_files}")
    print(f"Bad files: {len(bad_files)}") #checking if we have bad files while training
    
    #removing bad files in case we are stuck with some garbage files.
    if bad_files:
        print(f"Removing {len(bad_files)} bad files...")
        final_df = final_df[~final_df['feature_path'].isin(bad_files)]
        print(f"Remaining good files: {len(final_df)}")
    
    #debug message
    if len(final_df) == 0:
        print("\n❌ CRITICAL ERROR: No valid files found!")
        print("Please check:")
        print("1. Run the corrected feature_generator.py?")
        print("2. Check if .pt files in drone_dataset_spectrograms/ folder.")
        exit(1)
    
    # Split data
    print(f"\n=== SPLITTING DATA ===")
    
    # Create label mapping for 3 classes
    label_map = {'Mavic_1': 0, 'Mavic_2': 1, 'Mavic_Mini': 2}
    final_df['label_idx'] = final_df['label_type'].map(label_map)
    final_df['label_idx'] = final_df['label_idx'].fillna(0)  # Fill any NaN with 0
    
    train_df, val_df = train_test_split(
        final_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=final_df['label_idx']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Class balance
    print(f"\n=== CLASS BALANCE ===")
    class_counts = [0, 0, 0]
    class_names = ['Mavic_1', 'Mavic_2', 'Mavic_Mini']
    
    for i, name in enumerate(class_names):
        class_counts[i] = (train_df['label_type'] == name).sum()
        print(f"  {name}: {class_counts[i]} samples")
    
    # Create dataloaders
    print(f"\n=== CREATING DATALOADERS ===")
    train_dataset = DroneAcousticDataset(train_df, normalize=True)
    val_dataset = DroneAcousticDataset(val_df, normalize=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Create and train model
    print(f"\n=== SETTING UP MODEL ===")
    model = setup_model(device, class_counts)
    
    print(f"\n=== STARTING TRAINING ===")
    model = train_model(model, train_loader, val_loader, device, class_counts)
    
    # Final evaluation
    print(f"\n{'='*50}")
    print("FINAL EVALUATION")
    print('='*50)
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]
        confusion_matrix = np.zeros((3, 3), dtype=int)
        
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                class_total[true_label] += 1
                if true_label == pred_label:
                    class_correct[true_label] += 1
                confusion_matrix[true_label][pred_label] += 1
        
        overall_acc = 100 * correct / total
        class_accuracies = []
        for i in range(3):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0.0)
        
        balanced_acc = sum(class_accuracies) / 3
        
        print(f"\nOverall Accuracy: {overall_acc:.2f}%")
        print(f"Balanced Accuracy: {balanced_acc:.2f}%")
        print(f"\nPer-class Accuracy:")
        for i in range(3):
            print(f"  {class_names[i]}: {class_accuracies[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        print(f"\nConfusion Matrix:")
        print("True ↓ / Predicted →")
        print("       Mavic_1  Mavic_2  Mavic_Mini")
        for i in range(3):
            row = f"{class_names[i]:10}"
            for j in range(3):
                row += f"{confusion_matrix[i][j]:8d}"
            print(row)
    
    print("\n" + "="*50)
    print("EVALUATION:")
    if balanced_acc >= 80:
        print("✓ EXCELLENT: Model performs very well!")
    elif balanced_acc >= 70:
        print("✓ GOOD: Model performs well")
    elif balanced_acc >= 60:
        print("~ FAIR: Model needs improvement")
    else:
        print("✗ POOR: Significant improvement needed")