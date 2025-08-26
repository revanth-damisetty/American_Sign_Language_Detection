

# 🖐 Sign Language Recognition

A real-time sign language recognition system that leverages **computer vision, deep learning, and web technologies** to interpret hand gestures. Using **MediaPipe Hands**, the system accurately detects **21 3D hand landmarks** and preprocesses them to feed into a **Keras-based deep learning model** trained on a 3D tilt-aware dataset of sign language gestures. The application provides a **seamless real-time experience** via webcam, displaying both the predicted sign and its **confidence score** directly on the video feed.

Designed with **modularity and scalability** in mind, the project includes a **training notebook** for model experimentation, a **Flask API** for programmatic access, and a **real-time client script** for live demonstration. It can be extended to support **multi-hand recognition, mobile deployment, or integration with web and audio interfaces**, making it an ideal foundation for applications in **assistive technologies, accessibility tools, and human-computer interaction**.

Key highlights:

-   **Accurate gesture detection:** Captures subtle 3D hand movements with MediaPipe.
    
-   **Deep learning prediction:** Uses a tilt-aware Keras model for robust sign recognition.
    
-   **Real-time feedback:** Predictions are displayed instantly on live webcam feed.
    
-   **Flexible architecture:** Modular design supports training, API access, and real-time deployment.
    
-   **Extensible framework:** Can be enhanced for mobile/web use, multi-hand gestures, and additional sign languages.
    

----------

## 🚀 Features

-   Real-time gesture recognition via webcam
    
-   3D hand landmark extraction with MediaPipe
    
-   Scaled feature normalization for robust predictions
    
-   Flask API for prediction requests
    
-   Modular design: training, API, and real-time client
    
-   Confidence score for each prediction
    
-   Easy integration with other Python projects or web apps
    

----------

## 📁 Project Structure

```
SignLanguageRecognition/
│
├── TiltedModel/
│   ├── sign_model_3dtilt.h5       # Trained Keras model
│   ├── label_encoder.pkl           # Label encoder for sign classes
│   └── scaler.pkl                  # Scaler for feature normalization
│
├── app.py                          # Flask API for predictions
├── realtime.py                      # Real-time webcam client
├── titled_training.ipynb            # Notebook for model training
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies

```

----------

## ⚙️ Setup & Installation

### Clone the repository

```
git clone <repository_url>
cd SignLanguageRecognition

```

### Create a virtual environment

```
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

```

### Install dependencies

```
pip install -r requirements.txt

```

----------

## 🧠 Model Training

-   **Landmark Extraction:** MediaPipe Hands for 21 3D landmarks
    
-   **Preprocessing:** Flatten landmarks and scale using `scaler.pkl`
    
-   **Model:** Keras Sequential model trained to predict sign classes
    
-   **Saved Files:** `sign_model_3dtilt.h5`, `label_encoder.pkl`, `scaler.pkl`
    
-   **Notebook:** `titled_training.ipynb` for training, testing, and visualization
    

----------

## 🎥 Real-Time Prediction

-   Captures webcam frames every 1.5 seconds
    
-   Sends frames to Flask API (`/predict`)
    
-   Receives predicted sign + confidence
    
-   Displays prediction overlay on video feed
    

**Run real-time detection:**

```
python realtime.py

```

----------

## 🌐 API Usage

-   **Endpoint:** `POST /predict`
    
-   **Request:** Send an image file as `frame`
    
-   **Response:** JSON - Predicted letter with confidence value
    

**Example Response:**

```json
{
  "sign": "M",
  "confidence": 0.92
}

```

**Run API server:**

```
python app.py

```

----------

## 🧩 Methodology

-   **Hand Detection:** MediaPipe Hands (21 3D landmarks)
    
-   **Feature Flattening & Scaling:** Standardize input for model
    
-   **Prediction:** Feed into Keras model, choose highest probability class
    
-   **Display:** Overlay predicted sign + confidence on webcam feed
    

----------

## 🛠 Technologies

-   Python 3.x
    
-   OpenCV
    
-   MediaPipe
    
-   TensorFlow / Keras
    
-   Flask
    
-   NumPy, Pickle
    

----------

## 🔮 Future Improvements

-   Lightweight and faster models for mobile/web deployment
    
-   Web app interface with audio feedback
    
-   Multi-hand recognition
    
-   Support for multiple sign languages
    
-   Integration with voice-to-text for real-time conversation
    

----------

## 💡 Tips & Usage

-   Ensure good lighting for accurate hand landmark detection
    
-   Keep background simple to reduce detection errors
    
-   Run `realtime.py` only after starting the Flask API server
    
-   Use GPU if available for faster model inference
    

----------
