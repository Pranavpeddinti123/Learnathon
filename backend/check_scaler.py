import joblib
import os

scaler_path = 'backend/saved_models/scaler.joblib'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Scaler Mean:", scaler.mean_)
    print("Scaler Scale (Std):", scaler.scale_)
else:
    print("Scaler not found!")
