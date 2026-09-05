import sys
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/letter_model.keras"

print("Loading letter recognition model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


def recognize_letter(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("ERROR: drawing.png not found")
        return

    # Find the written area
    _, thresh = cv2.threshold(
        image,
        50,
        255,
        cv2.THRESH_BINARY
    )

    coords = cv2.findNonZero(thresh)

    if coords is None:
        print("No drawing detected!")
        return

    x, y, w, h = cv2.boundingRect(coords)

    # Crop the letter
    letter = image[y:y+h, x:x+w]

    # Make it square
    size = max(w, h)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = letter

    # Resize to EMNIST size
    square = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    square = square.astype("float32") / 255.0

    # Add channel + batch
    square = square.reshape(
        1, 28, 28, 1
    )

    # Prediction
    prediction = model.predict(
        square,
        verbose=0
    )

    predicted_class = np.argmax(prediction[0])

    confidence = np.max(prediction[0]) * 100

    # EMNIST Letters:
    # 0 = A
    # 1 = B
    # ...
    # 25 = Z

    letter = chr(
        ord("A") + predicted_class
    )

    print()
    print("==============================")
    print("     LETTER RECOGNITION")
    print("==============================")
    print(f"Detected Letter : {letter}")
    print(f"Confidence      : {confidence:.2f}%")
    print("==============================")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide image path")
    else:
        recognize_letter(sys.argv[1])