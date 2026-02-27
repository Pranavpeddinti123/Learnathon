# 🏃 Human Activity Recognition (HAR) System

An end-to-end deep learning system for classifying human activities from accelerometer and gyroscope sensor data.

## 🎯 Project Overview

This project implements a complete Human Activity Recognition system using:
- **Deep Learning**: PyTorch LSTM (2 layers, 128 hidden units)
- **Dataset**: UCI HAR Dataset (~10,000 samples, 6 activities)
- **Backend**: FastAPI REST API
- **Frontend**: React with modern UI/UX
- **Activities**: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying

## 📁 Project Structure

```
Learnathon_human/
├── backend/
│   ├── app/
│   │   └── main.py              # FastAPI application
│   ├── ml/
│   │   ├── dataset.py           # Data loading & preprocessing
│   │   ├── model.py             # LSTM model architecture
│   │   └── train.py             # Training script
│   ├── data/                    # Dataset storage (auto-downloaded)
│   ├── saved_models/            # Trained model checkpoints
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── FileUpload.jsx   # File upload interface
    │   │   ├── ManualInput.jsx  # Manual data entry
    │   │   └── Results.jsx      # Prediction visualization
    │   ├── App.jsx
    │   └── App.css
    └── package.json
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn
- ~500MB free disk space (for dataset + dependencies)

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset and train model**
   ```bash
   python ml/train.py
   ```
   
   This will:
   - Download UCI HAR Dataset (~60MB)
   - Train LSTM model (30 epochs, ~5-10 minutes on CPU)
   - Save best model to `saved_models/best_model.pth`
   - Generate training plots and classification report
   - Target accuracy: >85%

5. **Start FastAPI server**
   ```bash
   python -m uvicorn app.main:app --reload
   ```
   
   Backend will run on: `http://localhost:8000`
   - API docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```
   
   Frontend will run on: `http://localhost:5173` (or next available port)

## 🎮 Usage

### Option 1: File Upload
1. Prepare a sensor data file (`.txt` or `.csv`)
   - Format: 128 rows × 9 columns
   - Columns: body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z
   - Values: comma or space separated

2. Upload file via drag-and-drop or file picker
3. Click "Predict Activity"
4. View results with confidence scores

### Option 2: Manual Input
1. Click "Manual Input" tab
2. Generate random sample data or edit values manually
3. Click "Predict Activity"
4. View results with probability distribution

### Sample Data Location
You can find sample test data in `backend/data/UCI HAR Dataset/test/Inertial Signals/` after training.

## 📊 Model Details

### Architecture
```
Input: (batch, 128 timesteps, 9 features)
  ↓
LSTM Layer 1 (128 hidden units)
  ↓
LSTM Layer 2 (128 hidden units)
  ↓
Dropout (0.3)
  ↓
Fully Connected (128 → 6)
  ↓
Softmax
  ↓
Output: (batch, 6 classes)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Epochs**: 30
- **Batch Size**: 64
- **Expected Accuracy**: >85% on test set

### Dataset
- **Source**: UCI HAR Dataset
- **Training Samples**: 7,352
- **Test Samples**: 2,947
- **Sampling Rate**: 50Hz
- **Window Size**: 2.56 seconds (128 readings)

## 🔗 API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check and model status

### `GET /activities`
List of supported activities

### `POST /predict`
Predict activity from sensor data

**Request Body:**
```json
{
  "data": [[0.1, -0.2, 0.3, ...], ...] // 128×9 array
}
```

**Response:**
```json
{
  "activity": "WALKING",
  "confidence": 0.95,
  "probabilities": {
    "WALKING": 0.95,
    "WALKING_UPSTAIRS": 0.03,
    ...
  }
}
```

## 🐛 Troubleshooting

### Backend Issues

**Model not found error:**
```bash
# Make sure you've trained the model first
python backend/ml/train.py
```

**Port 8000 already in use:**
```bash
# Use a different port
python -m uvicorn app.main:app --reload --port 8001
# Update frontend API URL in App.jsx
```

**Dataset download fails:**
- Check internet connection
- Manually download from: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- Extract to `backend/data/`

### Frontend Issues

**CORS errors:**
- Ensure backend CORS middleware is configured correctly
- Backend must be running before making requests

**Connection refused:**
- Verify backend is running on `http://localhost:8000`
- Check firewall settings

## 📦 Technology Stack

### Backend
- **PyTorch**: Deep learning framework
- **FastAPI**: High-performance web framework
- **Uvicorn**: ASGI server
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Preprocessing and metrics
- **Matplotlib/Seaborn**: Visualization

### Frontend
- **React**: UI framework
- **Vite**: Build tool
- **CSS3**: Modern styling with gradients and animations

## 🎨 Features

✅ Real-time activity classification  
✅ Two input modes (file upload & manual entry)  
✅ Confidence scores and probability distribution  
✅ Modern, responsive dark-themed UI  
✅ Drag-and-drop file upload  
✅ Interactive data visualization  
✅ REST API with automatic documentation  
✅ Model training with visualization  

## 📝 Notes

- First run will download the dataset (~60MB)
- Training takes 5-10 minutes on CPU, 1-2 minutes on GPU
- Frontend connects to `localhost:8000` by default
- Model achieves >85% accuracy on test set

## 🔮 Future Enhancements

- [ ] Real-time sensor data streaming
- [ ] Mobile app integration
- [ ] Model deployment with Docker
- [ ] Additional activity classes
- [ ] Transfer learning capabilities
- [ ] Comparative model analysis

---

**Built for Learnathon | PyTorch LSTM + FastAPI + React**
