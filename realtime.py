import cv2
import requests
import time

print("🚀 Starting real-time sign language detection...")

url = "http://127.0.0.1:5000/predict"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("✅ Webcam opened successfully.")
print("🟢 Real-Time Sign Language Prediction (Press 'q' to quit)")

last_pred_time = 0
prediction_delay = 1.5  # seconds between predictions
text = "Waiting for prediction..."

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    current_time = time.time()
    if current_time - last_pred_time >= prediction_delay:
        # Save frame temporarily
        image_path = "temp_webcam.jpg"
        cv2.imwrite(image_path, frame)

        # Send to Flask API
        with open(image_path, "rb") as f:
            files = {'frame': f}
            try:
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    result = response.json()
                    sign = result['sign']
                    confidence = result['confidence']
                    text = f"{sign} ({confidence*100:.1f}%)"
                else:
                    text = "No hand detected"
            except requests.exceptions.RequestException:
                text = "API Error"

        last_pred_time = current_time

    # Show prediction on screen
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Sign Prediction", frame)

    # Maintain smooth video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

