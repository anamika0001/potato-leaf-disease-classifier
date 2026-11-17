# 🥔 Potato Leaf Disease Detector

An AI-powered web application that identifies common potato leaf diseases from uploaded images using a trained Convolutional Neural Network (CNN).

> [!NOTE]
> This project was built as a final-year AI/ML project. The CNN model was trained on a dataset of potato leaf images and classifies them into three categories: **Early Blight**, **Late Blight**, and **Healthy**.

---

## 📋 Problem Statement

Potato crops are susceptible to diseases like Early Blight and Late Blight, which can significantly reduce yield if not detected early. Manual inspection is time-consuming and requires expertise. This application provides a quick AI-based screening tool that farmers and agricultural students can use to get an initial assessment by simply uploading a photo of a potato leaf.

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| **Frontend** | Python, Streamlit |
| **Backend** | Python, FastAPI, Uvicorn |
| **ML Framework** | TensorFlow 2.22.0-rc0, Keras 3.16.0 |
| **Image Processing** | Pillow (PIL), NumPy |
| **PDF Reports** | ReportLab |
| **HTTP Client** | Python `requests` |
| **Testing** | pytest, httpx |

---

## 🧠 CNN Model Overview

The model is a custom **Sequential CNN** built with TensorFlow/Keras:

```
Input (128×128×3)
  → Rescaling (1/255)
  → Data Augmentation (flips, rotation, zoom, contrast)
  → Conv2D(32) → BatchNorm → ReLU → MaxPool
  → Conv2D(64) → BatchNorm → ReLU → MaxPool
  → Conv2D(128) → BatchNorm → ReLU → MaxPool
  → Conv2D(256) → BatchNorm → ReLU → MaxPool
  → GlobalAveragePooling2D
  → Dense(128, ReLU) → Dropout(0.3)
  → Dense(3, Softmax)
```

**Key details:**
- **Input**: 128×128 RGB images (pixel values 0–255; the model rescales internally)
- **Output**: 3 classes — `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`
- **Training data**: ~1,722 training images, ~430 validation images
- **Validation accuracy**: ~97%
- **Augmentation** is baked into the model but only active during training (`training=True`)

---

## 🏗️ Project Architecture

Yeh project **frontend-backend architecture** use karta hai, jisme Streamlit aur FastAPI alag-alag servers pe run karte hain:

```
User (Browser)
     │
     ▼
Streamlit Frontend (port 8501)
     │  HTTP POST /predict
     ▼
FastAPI Backend (port 8000)
     │
     ▼
Image Validation → Preprocessing → CNN Model → Prediction JSON
     │
     ▼
Streamlit shows results + PDF download
```

**Frontend (Streamlit)** sirf UI handle karta hai — image upload, preview, results display, aur PDF download. Yeh kabhi bhi directly model ko call nahi karta.

**Backend (FastAPI)** saara ML kaam karta hai — image validate karna, preprocess karna, model se prediction lena, aur JSON response bhejta hai.

---

## 📂 Folder Structure

```
potato-leaf-detector/
│
├── frontend/
│   ├── app.py              # Streamlit single-page application
│   └── styles.py           # Custom CSS styles
│
├── backend/
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI app with /health and /predict endpoints
│   ├── model_service.py     # Model loading and inference logic
│   ├── schemas.py           # Pydantic request/response schemas
│   └── disease_info.py      # Disease descriptions and guidance text
│
├── models/
│   ├── potato_cnn_model.keras   # Trained CNN model (saved Keras format)
│   └── class_names.txt          # Ordered class labels
│
├── utils/
│   ├── __init__.py          # Package marker
│   └── pdf_generator.py     # ReportLab PDF report generator
│
├── tests/
│   └── test_api.py          # Automated backend tests
│
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## 🔄 How It Works

### 1. Image Upload
User apni potato leaf ki photo (JPG/JPEG/PNG) Streamlit frontend pe upload karta hai.

### 2. Image Processing (Backend)
Jab user "Analyze Leaf" button click karta hai:
1. Frontend image bytes ko FastAPI backend pe HTTP POST `/predict` se bhejta hai
2. Backend file type, size (max 5 MB), aur image validity check karta hai
3. Image ko **RGB** me convert karke **128×128** me resize karta hai
4. NumPy array (float32) me convert karke batch dimension add karta hai → shape `(1, 128, 128, 3)`
5. **Pixel values 0–255 pe rehti hain** — model ke andar Rescaling layer khud 0–1 me convert karta hai

### 3. Model Prediction
- Model `training=False` mode me run hota hai (BatchNorm running stats use karta hai, Dropout inactive)
- Softmax layer teeno classes ke liye probability scores deta hai
- Sabse zyada score wali class ko `argmax` se select karte hain

### 4. Results Display
- Predicted disease name with colour badge (Green/Amber/Red)
- Confidence score percentage
- All three class scores as progress bars
- Disease description and precautions
- AI disclaimer

### 5. PDF Report
- ReportLab se ek professional PDF memory me generate hoti hai
- Usme image thumbnail, scores, description, precautions sab hota hai
- `st.download_button` se directly download hota hai — koi server storage nahi

---

## 🚀 Installation Instructions

### Prerequisites
- **Python 3.11+** (this project was developed with Python 3.14)
- **Windows** with VS Code (or any terminal)

### Step 1: Navigate to the project

```powershell
cd c:\Users\yadav\OneDrive\Desktop\DeepLearning\potato-leaf-detector
```

### Step 2: Use the existing virtual environment

The project uses the parent directory's `.venv`:

```powershell
cd c:\Users\yadav\OneDrive\Desktop\DeepLearning
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
pip install -r potato-leaf-detector\requirements.txt
```

> [!TIP]
> TensorFlow aur Keras pehle se installed hain `.venv` me. Agar fresh environment bana rahe ho toh `tensorflow>=2.16.0` bhi install karo.

---

## ▶️ Running the Application

You need **two terminal windows** — one for the backend, one for the frontend.

### Terminal 1 — Start FastAPI Backend

```powershell
cd c:\Users\yadav\OneDrive\Desktop\DeepLearning
.\.venv\Scripts\Activate.ps1
cd potato-leaf-detector
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Backend will be available at: `http://127.0.0.1:8000`
API docs (Swagger): `http://127.0.0.1:8000/docs`

### Terminal 2 — Start Streamlit Frontend

```powershell
cd c:\Users\yadav\OneDrive\Desktop\DeepLearning
.\.venv\Scripts\Activate.ps1
cd potato-leaf-detector
python -m streamlit run frontend/app.py
```

Frontend will open at: `http://localhost:8501`

> [!IMPORTANT]
> **Pehle backend start karo, phir frontend.** Agar backend nahi chal raha toh frontend pe "Unable to connect to the prediction service" error aayega.

---

## 🧪 Testing

### Run automated tests

```powershell
cd c:\Users\yadav\OneDrive\Desktop\DeepLearning\potato-leaf-detector
..\.venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

Tests cover:
- ✅ Health endpoint
- ✅ Valid image prediction (mocked model)
- ✅ Corrupted image rejection
- ✅ Unsupported format rejection
- ✅ Oversized file rejection
- ✅ Response schema validation
- ✅ Correct class order
- ✅ Model-not-loaded → 503 response

### Manual testing
1. Start both servers (backend + frontend)
2. Open `http://localhost:8501` in your browser
3. Upload any potato leaf image (JPG/PNG)
4. Click "Analyze Leaf"
5. Check prediction results, scores, and disease info
6. Try downloading the PDF report
7. Click "Analyze Another Image" to reset

---

## ❗ Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "Unable to connect to the prediction service" | FastAPI backend is not running | Start the backend first with `uvicorn` command |
| "Model could not be loaded" | `potato_cnn_model.keras` missing from `models/` | Run the training script to generate the model |
| `ModuleNotFoundError: No module named 'backend'` | Running from wrong directory | Make sure you `cd` into `potato-leaf-detector/` before running |
| Port 8000 already in use | Another process on that port | Kill the process or use `--port 8001` |
| Port 8501 already in use | Another Streamlit app running | Close the other Streamlit tab/process |
| `ImportError: tensorflow` | TF not installed | `pip install tensorflow` in the venv |

---

## 🛑 How to Stop and Restart

### Stop
- Press **Ctrl+C** in each terminal window to stop the servers

### Restart
- Simply run the `uvicorn` and `streamlit` commands again

---

## 🔮 Future Deployment Options

Agar aap is project ko internet pe deploy karna chahte ho:

1. **Streamlit Cloud** — Free hosting for Streamlit apps (frontend only; backend ko alag host karna padega)
2. **Railway / Render** — Free tier pe FastAPI backend deploy kar sakte ho
3. **Docker** — Dono services ko containerize karke ek saath deploy kar sakte ho
4. **Google Cloud Run** — Serverless deployment for both services
5. **AWS EC2 / Azure VM** — Full control wala deployment

> [!CAUTION]
> Production deployment ke liye CORS settings tighten karo, HTTPS enable karo, aur proper logging/monitoring setup karo. Current setup sirf local development ke liye hai.

---

## 📄 License

This project is for educational purposes.

---

*Built with ❤️ using TensorFlow, FastAPI, and Streamlit*
