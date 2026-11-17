# Potato Leaf Disease Classifier

**CNN-powered potato leaf disease screening with a Streamlit interface, FastAPI inference service, and downloadable PDF reports.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

## Overview

This project demonstrates an end-to-end image classification workflow for identifying common diseases in potato leaves. A custom Convolutional Neural Network (CNN), trained with TensorFlow/Keras, classifies an uploaded leaf image into one of three categories:

- **Early Blight** (`Potato___Early_blight`)
- **Late Blight** (`Potato___Late_blight`)
- **Healthy** (`Potato___healthy`)

The application provides predicted class scores, supporting disease information, and a downloadable PDF summary. It is intended as an **educational prototype and preliminary screening aid**, not a substitute for laboratory testing or expert agricultural diagnosis.

## Problem Statement

Potato crops are vulnerable to leaf diseases that can affect plant health and crop yield. Visual identification may require experience and can be difficult when symptoms overlap. This project explores how deep learning can assist with preliminary classification from leaf photographs through an accessible web interface.

## Key Features

- **Image upload:** JPG, JPEG, and PNG images, with server-side validation and a 5 MB upload limit.
- **CNN inference:** Classification into Early Blight, Late Blight, or Healthy.
- **Prediction transparency:** Display of the selected class and all three model probability scores.
- **Informational guidance:** Disease descriptions and general precautions.
- **PDF export:** Downloadable report containing the image, predicted class, scores, and guidance.
- **Separation of concerns:** Streamlit user interface and FastAPI inference service.
- **API tests:** Automated checks for endpoint behavior, input validation, and response structure.

## Technology Stack

| Layer | Technologies |
| --- | --- |
| Frontend | Python, Streamlit, custom CSS |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Deep learning | TensorFlow, Keras |
| Image processing | Pillow, NumPy |
| API communication | Requests |
| Report generation | ReportLab |
| Testing | pytest, httpx |

> The original development environment used TensorFlow 2.22.0-rc0 and Keras 3.16.0. Check `requirements.txt` and package compatibility when installing on another Python version or deployment platform.

## System Architecture

```text
                   User / Browser
                          |
                  Upload leaf image
                          v
                Streamlit Frontend
                  (localhost:8501)
                          |
                    HTTP POST
                     /predict
                          v
                   FastAPI API
                  (127.0.0.1:8000)
                          |
              Validate uploaded image
                          |
               Convert to RGB, resize
                    to 128 x 128
                          |
              Load trained Keras CNN
                          |
               Predict class scores
                          |
                   JSON response
                          v
                Streamlit Frontend
                          |
          Prediction + guidance + PDF
```

The frontend handles user interaction and presentation. The backend validates images, preprocesses inputs, performs model inference, and returns structured prediction results. The two services run separately in the current local-development configuration.

## CNN Architecture

The project uses a custom CNN with four convolutional blocks:

```text
Input: RGB image (128 x 128 x 3)
   |
Rescaling (1/255)
   |
Data augmentation (training only)
   |
Conv2D(32)  -> BatchNorm -> ReLU -> MaxPooling
   |
Conv2D(64)  -> BatchNorm -> ReLU -> MaxPooling
   |
Conv2D(128) -> BatchNorm -> ReLU -> MaxPooling
   |
Conv2D(256) -> BatchNorm -> ReLU -> MaxPooling
   |
GlobalAveragePooling2D
   |
Dense(128, ReLU) -> Dropout(0.3)
   |
Dense(3, Softmax)
   |
Early Blight / Late Blight / Healthy
```

### Input processing and inference

1. The backend checks file type, file size, and whether the upload is a readable image.
2. The image is converted to RGB and resized to **128 × 128** pixels.
3. Pixel data is converted into a `float32` NumPy array with shape **`(1, 128, 128, 3)`**.
4. Input values remain on the **0–255** scale because the model contains its own rescaling layer.
5. The CNN runs in inference mode (`training=False`); training-only augmentation and dropout are disabled.
6. The Softmax output provides three class scores. `argmax` selects the highest-scoring class.

**Important:** Softmax confidence is a model score, not a guarantee that the diagnosis is correct.

## Dataset and Evaluation

The documented dataset contains **2,152 potato leaf images**, split into approximately **1,722 training** and **430 validation** images (80/20 split).

| Class | Total images |
| --- | ---: |
| Early Blight | 1,000 |
| Late Blight | 1,000 |
| Healthy | 152 |
| **Total** | **2,152** |

The original project documentation reports **approximately 97% validation accuracy**. This figure refers to the validation split, **not an independent held-out test set**. Results may differ on real-world images because of lighting, backgrounds, image quality, class imbalance, and distribution shifts. No independently measured field-performance claim is made here.

The Healthy class is less represented than the disease classes. This imbalance should be considered when interpreting overall accuracy and future per-class evaluation.

## Project Structure

```text
potato-leaf-detector/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI endpoints
│   ├── model_service.py        # Model loading and inference
│   ├── schemas.py              # Response schemas
│   └── disease_info.py         # Disease information
├── frontend/
│   ├── app.py                  # Streamlit application
│   └── styles.py               # Custom styling
├── models/
│   ├── potato_cnn_model.keras  # Saved trained model
│   └── class_names.txt         # Ordered class labels
├── utils/
│   ├── __init__.py
│   └── pdf_generator.py        # PDF report generation
├── tests/
│   └── test_api.py             # API tests
├── .env.example                # Environment configuration example
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python and `pip` (use a Python version compatible with the packages in `requirements.txt`)
- Git
- A terminal (PowerShell examples are provided below)

### 1. Clone the repository

```powershell
git clone https://github.com/anamika0001/potato-leaf-disease-classifier.git
cd potato-leaf-disease-classifier
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it using `source .venv/bin/activate`.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installation fails, verify the TensorFlow/Python version combination and the dependencies specified in `requirements.txt`.

### 4. Start the backend

Open a terminal in the repository root, activate the environment, and run:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

### 5. Start the frontend

In a **second terminal**, activate the same environment from the repository root:

```powershell
python -m streamlit run frontend/app.py
```

Open **`http://localhost:8501`** in your browser. Keep the FastAPI service running while using the frontend.

## Using the Application

1. Upload a potato leaf image in JPG, JPEG, or PNG format.
2. Select **Analyze Leaf**.
3. Review the predicted class, class scores, and disease information.
4. Optionally download the PDF report.
5. Select **Analyze Another Image** to start a new analysis.

## API and Testing

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Check API health |
| `/predict` | `POST` | Validate an uploaded image and return prediction results |
| `/docs` | `GET` | View FastAPI's interactive API documentation |

Run the automated API tests from the repository root:

```powershell
python -m pytest tests/test_api.py -v
```

The test suite is designed to cover health checks, valid image requests with mocked inference, unsupported or corrupted images, oversized uploads, response validation, class ordering, and model-unavailable behavior. **Mocked API tests do not independently establish CNN prediction accuracy.**

## Troubleshooting

| Issue | Suggested check |
| --- | --- |
| Frontend cannot connect to the prediction service | Confirm FastAPI is running and that the frontend is configured to use its correct URL. |
| Model fails to load | Confirm `models/potato_cnn_model.keras` exists and TensorFlow/Keras versions are compatible. |
| `ModuleNotFoundError: backend` | Run the backend command from the repository root. |
| Port 8000 or 8501 is already in use | Stop the conflicting process or configure an available port. |
| TensorFlow import fails | Activate the correct environment and install compatible dependencies. |

Stop either service with **Ctrl+C** in its terminal.

## Deployment Considerations

The repository currently documents a **two-service architecture**. Hosting the Streamlit frontend alone does not automatically host the FastAPI inference backend. A public deployment requires a reachable backend URL, appropriate frontend configuration, compatible runtime dependencies, and security settings such as HTTPS, restricted CORS, and request-size limits.

Potential hosting approaches include deploying the frontend and backend separately or packaging both services for a suitable container-based platform. **No live deployment URL is provided in this repository documentation.**

## Limitations and Future Work

- Evaluate the model on a separate test set and report per-class precision, recall, F1-score, and a confusion matrix.
- Improve class balance and test performance on images captured outside the training dataset.
- Add model versioning, reproducible training artifacts, and inference monitoring.
- Evaluate robustness to lighting, leaf orientation, image backgrounds, and unfamiliar symptoms.
- Deploy and validate the full frontend–backend workflow in a hosted environment.

## Project Status

**Educational prototype.** The repository contains the application source code and a saved CNN model. Local execution and deployment should be verified in the target environment before describing the application as production-ready.

## Author

**Anamika Yadav**  
GitHub: [@anamika0001](https://github.com/anamika0001)

## Usage Notice

Developed for educational and demonstration purposes. The classification output should not be treated as a definitive plant disease diagnosis. The statement of educational use is not an open-source license; reuse permissions should be specified separately if a license is added.
