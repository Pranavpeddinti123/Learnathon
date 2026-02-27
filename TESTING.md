# 🎯 Testing Your HAR Application

## ✅ Current Status

**System is fast and feature-rich!**
- 🟢 **Backend API**: http://127.0.0.1:8000
- 🟢 **Frontend UI**: http://localhost:5173
- 🤖 **Sensor Model**: LSTM (90%+ accuracy)
- 👁️ **Vision Model**: CLIP (Zero-shot classification)

---

## 🧪 How to Test the Application

### 1️⃣ Test Image Recognition (NEW! 📸)

1. Click on the **"🖼️ Image Input"** tab.
2. Drag & drop an image of a person:
   - 🏃 **Running/Walking**
   - 🪑 **Sitting**
   - 🛌 **Laying down**
3. Click **"👁️ Predict from Image"**.
4. The AI will analyze the visual content and predict the activity!

### 2️⃣ Test Manual Input (Improved!)

1. Click on the **"✏️ Manual Input"** tab.
2. Use the **New Pattern Buttons**:
   - **🚶 Walking**: Generates rhythmic, high-variance data.
   - **🧍 Standing**: Generates steady, low-variance data.
   - **🛌 Laying**: Generates steady data with different gravity axis.
   - **🎲 Random**: Generates pure noise (usually predicts Sitting/Standing).
3. Click **"🔮 Predict Activity"**.

### 3️⃣ Test File Upload

1. Click on the **"📁 File Upload"** tab.
2. Upload a text file with 128 rows × 9 columns.
   - *Tip: Use the generated data from Manual Input to create a test file if you don't have one.*

---

## 🔍 What to Expect

| Feature | Input | Expected Output |
| :--- | :--- | :--- |
| **Image** | Photo of runner | **WALKING** (High confidence) |
| **Image** | Photo of person at desk | **SITTING** |
| **Manual** | "Walking" Pattern | **WALKING** |
| **Manual** | "Standing" Pattern | **STANDING** |

---

## 🛠️ Troubleshooting

### "Invalid data format"
- Make sure you are uploading **Images** to the Image tab and **Text Files** to the File Upload tab.

### Image Prediction is Slow
- The first time you run an image prediction, the backend downloads the CLIP model (~500MB). This happens only once. Subsequent requests will be fast.

### Backend Error 500
- Check your backend terminal. If you see "ModuleNotFoundError", ensure you restarted the backend after we installed `transformers`.

---

**Enjoy your Multi-Modal Activity Recognition System! 🚀**
