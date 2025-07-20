# 🇺🇸 American Sign Language Detection 📱

An end-to-end deep-learning project to **detect American Sign Language (ASL) gestures**, supporting **29 classes** of alphabets/gestures using **MobileNetV2**, and deployed on smartphones via **TensorFlow Lite**.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)  
2. [Architecture & Model](#architecture--model)  
3. [How It Works](#how-it-works)  
4. [Directory Structure](#directory-structure)  
5. [Setup & Installation](#setup--installation)  
6. [Usage](#usage)  
7. [Demo & Screenshots](#demo--screenshots)  
8. [Results & Performance](#results--performance)  
9. [Future Work](#future-work)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## 🔍 Project Overview

This project uses a deep-learning model to recognize ASL alphabets through gesture recognition:

- **Dataset:** A curated MOD dataset featuring 29 ASL classes.  
- **Model:** Utilizes **MobileNetV2**, pre-trained then fine-tuned via **TensorFlow** / **Keras**.  
- **Mobile Deployment:** Converted trained model into **TensorFlow Lite** (.tflite) format for on-device inference.  
- **Interface:** Integrated into a **Java-based mobile app**.

---

## 🧱 Architecture & Model

```
+----------------+      +------------------+      +------------------------+
| ASL Image Feed | ---> | MobileNetV2      | ---> | Dense (29-softmax)     |
|  (camera/app)  |      | + Custom layers  |      |                        |
+----------------+      +------------------+      +------------------------+
                                   ↓
                             TF SavedModel
                                   ↓
                              TF Lite (.tflite)
                                   ↓
                            Mobile Deployment
```

The core **MobileNetV2** serves as a lightweight backbone ideal for mobile deployment. A final dense layer is added to match the 29 target classes.

---

## ⚙️ How It Works

1. **Data Preprocessing:**  
   - Images resized to `224×224`, normalized for MobileNetV2.  
   - Labels encoded into one-hot vectors.

2. **Model Training:**  
   - MobileNetV2 loaded with ImageNet weights.  
   - Fine-tuned on the ASL dataset until validation metrics plateau.

3. **Conversion to TFLite:**  
   - Model saved as `model.tflite` using `tf.lite.TFLiteConverter`.  
   - Optimizations (quantization) applied for faster mobile inference.

4. **Deployment in App:**  
   - Model integrated into an Android app (Java/Kotlin).  
   - Real-time inference on camera feed, mapping gestures to ASL alphabets.

---

## 🗂️ Directory Structure

```
/
├── notebook/           # Jupyter notebooks: model training, evaluation, TFLite conversion
├── script/             # Utility scripts for conversion, dataset management
├── ASL App/            # Android mobile app integration with TFLite
├── results/            # Training logs, accuracy/loss plots, confusion matrix
├── model/              # SavedModel & TFLite .tflite files
├── README.md           # Project documentation
├── LICENSE             # MIT License
└── CODE_OF_CONDUCT.md
```

---

## 🛠️ Setup & Installation

Clone the repo and set up Python environment:

```bash
git clone https://github.com/revanth-damisetty/American_Sign_Language_Detection.git
cd American_Sign_Language_Detection

# For training (in notebook)
pip install -r requirements.txt  # includes TensorFlow, Keras, OpenCV, etc.

# For mobile app
Open Android Studio → Import ASL App folder
Add the model.tflite to the assets directory
Ensure camera & storage permissions are set
```

---

## ▶️ Usage

### 1. Train & Convert

- Open `notebook/ASL_train.ipynb`: run cells to preprocess data, train, visualize metrics.  
- After training, convert the model:  

```python
import tensorflow as tf

model = tf.keras.models.load_model('model/saved_model/')
converter = tf.lite.TFLiteConverter.from_saved_model('model/saved_model/')
tflite_model = converter.convert()

with open('model/asl_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 2. Deploy on Mobile

- Place `asl_model.tflite` into `ASL App/app/src/main/assets/`.  
- Run the Android app on a device/simulator.  
- The app displays live camera feed with recognized ASL letter overlay.

---

## 📸 Demo & Screenshots

*Insert screenshots showing live ASL detection from the app here*

---

## 📊 Results & Performance

- Model trained to over **90% accuracy** on test set (metrics logged in `results/`).  
- **Loss & accuracy curves, confusion matrix**, all visualized in `results/` folder.  
- Model size: ~**X MB**, optimized for mobile.

---

## 🚀 Future Work

- **Text-to-Speech:** Convert predicted letters into spoken words.  
- **Sentence tracking:** Recognize continuous ASL sentences, not just isolated letters.  
- **Extended gestures:** Include digits, common phrases, and full vocabulary.

---

## 🤝 Contributing

1. Fork the repo  
2. Create a new branch: `git checkout -b feature-xyz`  
3. Commit changes & push  
4. Open a Pull Request  

Please adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 💡 Acknowledgements

- Based on **MobileNetV2** architecture, fine-tuned for ASL.  
- Deployed via **TensorFlow Lite** for mobile inference.  
- Kudos to the original repo's author for the core pipeline.

---
