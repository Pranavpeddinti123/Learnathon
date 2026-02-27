import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import load_data, preprocess_data, HARDataset, ACTIVITY_LABELS
from ml.model import get_model, count_parameters


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='backend/saved_models'):
    """Plot training history"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {os.path.join(save_path, 'training_history.png')}")
    plt.close()


def evaluate_model(model, dataloader, device):
    """Detailed model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, save_path='backend/saved_models'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ACTIVITY_LABELS.values(),
                yticklabels=ACTIVITY_LABELS.values())
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {os.path.join(save_path, 'confusion_matrix.png')}")
    plt.close()


def train_model(num_epochs=30, batch_size=64, learning_rate=0.001, 
                hidden_size=128, num_layers=2, dropout=0.3):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\n" + "="*50)
    print("Loading and preprocessing data...")
    print("="*50)
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test, scaler = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Save the scaler
    os.makedirs('backend/saved_models', exist_ok=True)
    joblib.dump(scaler, 'backend/saved_models/scaler.joblib')
    print(f"Scaler saved to backend/saved_models/scaler.joblib")
    
    # Create datasets and dataloaders
    train_dataset = HARDataset(X_train, y_train)
    test_dataset = HARDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    model = get_model(
        input_size=9,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=6,
        dropout=dropout
    )
    model = model.to(device)
    
    print(model)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('backend/saved_models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'backend/saved_models/best_model.pth')
            print(f"  → Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Plot training history
    print("\n" + "="*50)
    print("Generating training plots...")
    print("="*50)
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model and evaluate
    print("\n" + "="*50)
    print("Evaluating best model...")
    print("="*50)
    checkpoint = torch.load('backend/saved_models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    y_pred, y_true = evaluate_model(model, test_loader, device)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=ACTIVITY_LABELS.values()))
    
    # Save classification report
    with open('backend/saved_models/classification_report.txt', 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=ACTIVITY_LABELS.values()))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print("="*50)
    
    return model, best_val_acc


if __name__ == '__main__':
    # Train model
    model, best_acc = train_model(
        num_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
