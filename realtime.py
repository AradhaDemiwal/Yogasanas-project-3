import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("yoga_model.h5")

# 👉 IMPORTANT: apne class labels yahan likho (same order as training)
class_names = [
    'balasana_average',
    'balasana_good',
    'balasana_poor',
    'bhujangasana_average',
    'bhujangasana_good',
    'bhujangasana_poor',
    'padmasana_average',
    'padmasana_good',
    'padmasana_poor',
    'parvatasana_average',
    'parvatasana_good',
    'parvatasana_poor',
    'tadasana_average',
    'tadasana_good',
    'tadasana_poor',
    'trikonasana_average',
    'trikonasana_good',
    'trikonasana_poor',
    'vrikshasana_average',
    'vrikshasana_good',
    'vrikshasana_poor'
]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize image
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    # Prediction
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)

    label = class_names[class_index]

    # Suggestion logic
    if "poor" in label:
        msg = "Improve posture!"
        color = (0, 0, 255)
    elif "average" in label:
        msg = "Almost correct"
        color = (0, 255, 255)
    else:
        msg = "Perfect pose!"
        color = (0, 255, 0)

    # Display pose name
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    # Display suggestion
    cv2.putText(frame, msg, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    # Show window
    cv2.imshow("Yoga Pose Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()