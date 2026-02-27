import requests
import json
import numpy as np

def generate_walking_data():
    timesteps = 128
    freq = 2.0
    amp = 0.3
    t = np.linspace(0, (timesteps-1)/50, timesteps)
    
    # Base means for walking
    body_acc_mean = [-0.0152, -0.0080, 0.0038]
    gyro_mean = [-0.2585, 0.1394, -0.0020]
    total_acc_mean = [0.9711, -0.3369, 0.0176]
    
    data = []
    for i in range(timesteps):
        row = []
        phase = 0.5
        wave = np.sin(2 * np.pi * t[i] * freq + phase)
        g_wave = np.cos(2 * np.pi * t[i] * freq + phase)
        
        # Body Acc
        row.append(float(body_acc_mean[0] + wave * amp))
        row.append(float(body_acc_mean[1] + g_wave * amp))
        row.append(float(body_acc_mean[2] + wave * amp * 0.5))
        
        # Gyro
        row.append(float(gyro_mean[0] + wave * 1.5))
        row.append(float(gyro_mean[1] + g_wave * 1.0))
        row.append(float(gyro_mean[2] + wave * 0.5))
        
        # Total Acc
        row.append(float(total_acc_mean[0] + wave * amp))
        row.append(float(total_acc_mean[1] + g_wave * amp))
        row.append(float(total_acc_mean[2] + wave * amp * 0.5))
        
        data.append(row)
    
    return data

def test_prediction(data):
    url = "http://localhost:8000/predict"
    payload = {"data": data}
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    walking_data = generate_walking_data()
    # Save to CSV for user to download if needed
    with open("walking_test_sample.csv", "w") as f:
        for row in walking_data:
            f.write(",".join(map(str, row)) + "\n")
    
    result = test_prediction(walking_data)
    print("Prediction Result:")
    print(json.dumps(result, indent=2))
