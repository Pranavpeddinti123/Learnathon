import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.dataset import load_data, ACTIVITY_LABELS

def check_samples():
    X_train, y_train, X_test, y_test = load_data()
    
    with open('backend/data_report.txt', 'w') as f:
        for i in range(6):
            indices = np.where(y_test == i)[0]
            if len(indices) == 0:
                continue
            idx = indices[0]
            label = ACTIVITY_LABELS[i]
            f.write(f"\n--- {label} (Label {i}) ---\n")
            # Features: [body_acc_x, body_acc_y, body_acc_z, gyro_x, gyro_y, gyro_z, total_acc_x, total_acc_y, total_acc_z]
            sample = X_test[idx] # (128, 9)
            means = np.mean(sample, axis=0)
            stds = np.std(sample, axis=0)
            mins = np.min(sample, axis=0)
            maxs = np.max(sample, axis=0)
            
            f.write(f"Means:\n")
            f.write(f"  Body Acc:  {means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f}\n")
            f.write(f"  Gyro:      {means[3]:.4f}, {means[4]:.4f}, {means[5]:.4f}\n")
            f.write(f"  Total Acc: {means[6]:.4f}, {means[7]:.4f}, {means[8]:.4f}\n")
            f.write(f"Ranges (Min, Max):\n")
            f.write(f"  Total Acc X: {mins[6]:.4f}, {maxs[6]:.4f}\n")
            f.write(f"  Total Acc Y: {mins[7]:.4f}, {maxs[7]:.4f}\n")
            f.write(f"  Total Acc Z: {mins[8]:.4f}, {maxs[8]:.4f}\n")
            
            f.write(f"First 5 rows (raw):\n")
            for row in sample[:5]:
                f.write(f"  {row.tolist()}\n")

if __name__ == '__main__':
    check_samples()
