import cv2
from keras.models import model_from_json
import numpy as np

# Load the trained model
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    """Preprocess the image for model prediction."""
    feature = np.array(image, dtype="float32")  # Ensure correct data type
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0  # Normalize pixel values

# Open webcam and set resolution to 1080p
webcam = cv2.VideoCapture(0)
webcam.set(3, 1920)  # Width
webcam.set(4, 1080)  # Height

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image")
        break

    # Flip the image horizontally to correct mirroring
    im = cv2.flip(im, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        try:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)

            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Ensure text is drawn within the frame
            text_x = max(10, p - 10)
            text_y = max(20, q - 10)

            cv2.putText(im, prediction_label, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        except Exception as e:
            print("Error processing face:", e)

    # Resize the window to half the screen width (960px)
    im_resized = cv2.resize(im, (960, 540))  # Half-width, keeping aspect ratio

    cv2.imshow("Output", im_resized)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
