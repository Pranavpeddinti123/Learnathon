# Quick Start Guide

## Backend Setup and Training

1. **Setup Python environment**
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python ml/train.py
   ```
   Wait for training to complete (~5-10 minutes). You should see accuracy >85%.

3. **Start the API server**
   ```bash
   python -m uvicorn app.main:app --reload
   ```
   Keep this terminal open. Backend will run on http://localhost:8000

## Frontend Setup

1. **In a new terminal, setup React**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the frontend**
   ```bash
   npm run dev
   ```
   Frontend will run on http://localhost:5173

3. **Open your browser**
   Go to http://localhost:5173 and try the app!

## Testing the System

### Quick Test with Manual Input
1. Click "Manual Input" tab
2. Click "Generate Random Data"
3. Click "Predict Activity"
4. See the predicted activity!

### Test with File
Sample test files are in `backend/data/UCI HAR Dataset/test/Inertial Signals/` after training.

## Common Commands

**Backend:**
- Train model: `python backend/ml/train.py`
- Start API: `python -m uvicorn app.main:app --reload` (from backend/)
- View API docs: http://localhost:8000/docs

**Frontend:**
- Install deps: `npm install` (from frontend/)
- Start dev server: `npm run dev`
- Build for production: `npm run build`
