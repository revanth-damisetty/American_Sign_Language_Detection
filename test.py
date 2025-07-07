import cv2
import requests

# Step 1: Capture a frame from webcam
cap = cv2.VideoCapture(0)
print("📸 Capturing image from webcam...")
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to capture image.")
    exit()

# Step 2: Save the frame to a temporary image file
image_path = "temp_webcam.jpg"
cv2.imwrite(image_path, frame)
print(f"✅ Frame saved to {image_path}")

# Step 3: Send it to the Flask API
url = "http://127.0.0.1:5000/predict"
with open(image_path, "rb") as f:
    files = {'frame': f}
    response = requests.post(url, files=files)

# Step 4: Handle the response
if response.status_code == 200:
    result = response.json()
    print(f"🧠 Prediction: {result['sign']}, Confidence: {result['confidence']:.2f}")
else:
    print(f"❌ Error: {response.status_code}, {response.text}")
