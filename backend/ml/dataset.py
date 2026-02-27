import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler


class HARDataset(Dataset):
    """PyTorch Dataset for Human Activity Recognition"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def download_data(data_dir='backend/data'):
    """Download and extract UCI HAR Dataset"""
    
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    zip_path = os.path.join(data_dir, 'UCI_HAR_Dataset.zip')
    extract_path = data_dir
    
    # Check if already downloaded
    if os.path.exists(os.path.join(data_dir, 'UCI HAR Dataset')):
        print("Dataset already exists. Skipping download.")
        return os.path.join(data_dir, 'UCI HAR Dataset')
    
    print(f"Downloading UCI HAR Dataset (~60MB)...")
    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print(f"Dataset extracted to {extract_path}")
        os.remove(zip_path)  # Clean up zip file
        
        return os.path.join(data_dir, 'UCI HAR Dataset')
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please manually download from: {dataset_url}")
        raise


def load_signals(data_path, data_type='train'):
    """Load all signal data from the dataset"""
    
    signals_path = os.path.join(data_path, data_type, 'Inertial Signals')
    
    # Signal file names
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    
    signals = []
    
    for signal_type in signal_types:
        filename = os.path.join(signals_path, f'{signal_type}_{data_type}.txt')
        signal_data = pd.read_csv(filename, delim_whitespace=True, header=None)
        signals.append(signal_data.values)
    
    # Stack signals: (samples, timesteps, features)
    # Each signal is (samples, 128 timesteps)
    # We have 9 signals, so final shape: (samples, 128, 9)
    X = np.stack(signals, axis=-1)
    
    return X


def load_labels(data_path, data_type='train'):
    """Load activity labels"""
    
    labels_path = os.path.join(data_path, data_type, f'y_{data_type}.txt')
    y = pd.read_csv(labels_path, header=None, names=['activity'])
    
    # Convert labels to 0-indexed (originally 1-6, we need 0-5)
    y = y['activity'].values - 1
    
    return y


def load_data(data_dir='backend/data'):
    """Load complete train and test datasets"""
    
    dataset_path = os.path.join(data_dir, 'UCI HAR Dataset')
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Downloading...")
        download_data(data_dir)
    
    print("Loading training data...")
    X_train = load_signals(dataset_path, 'train')
    y_train = load_labels(dataset_path, 'train')
    
    print("Loading test data...")
    X_test = load_signals(dataset_path, 'test')
    y_test = load_labels(dataset_path, 'test')
    
    print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, y_train, X_test, y_test):
    """Normalize and prepare data for PyTorch"""
    
    # Get shapes
    n_train, timesteps, n_features = X_train.shape
    n_test = X_test.shape[0]
    
    # Reshape for scaling: (samples * timesteps, features)
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back to (samples, timesteps, features)
    X_train_scaled = X_train_scaled.reshape(n_train, timesteps, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test, timesteps, n_features)
    
    print("Data preprocessing complete.")
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


# Activity labels mapping
ACTIVITY_LABELS = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}


if __name__ == '__main__':
    # Test data loading
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test, scaler = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Create datasets
    train_dataset = HARDataset(X_train, y_train)
    test_dataset = HARDataset(X_test, y_test)
    
    print(f"\nDataset created successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
