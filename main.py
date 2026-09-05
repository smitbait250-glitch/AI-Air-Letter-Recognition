import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os


# ============================================================
# LOAD AI MODEL
# ============================================================

print("Loading letter recognition model...")

MODEL_PATH = "models/letter_model.keras"

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found:", MODEL_PATH)
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

print("Letter model loaded!")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not open")
    exit()

print("Camera started")
print("Move your INDEX FINGER to draw")
print("R = recognize letter")
print("C = clear")
print("Q = quit")


# ============================================================
# CANVAS
# ============================================================

canvas = np.zeros(
    (480, 640),
    dtype=np.uint8
)

previous_point = None

detected_letter = "-"
confidence = 0.0


# ============================================================
# RECOGNITION
# ============================================================

def recognize_letter(canvas):

    coords = cv2.findNonZero(canvas)

    if coords is None:
        return "-", 0.0

    x, y, w, h = cv2.boundingRect(coords)

    if w < 20 or h < 20:
        return "-", 0.0

    # --------------------------------------------------------
    # Crop the letter
    # --------------------------------------------------------

    letter = canvas[y:y+h, x:x+w]


    # --------------------------------------------------------
    # Add small padding
    # --------------------------------------------------------

    padding = 10

    padded = cv2.copyMakeBorder(
        letter,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0
    )


    # --------------------------------------------------------
    # Make square while keeping aspect ratio
    # --------------------------------------------------------

    h2, w2 = padded.shape

    size = max(h2, w2)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    xoff = (size - w2) // 2
    yoff = (size - h2) // 2

    square[
        yoff:yoff+h2,
        xoff:xoff+w2
    ] = padded


    # --------------------------------------------------------
    # Resize to EMNIST size
    # --------------------------------------------------------

    square = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )


    # ========================================================
    # IMPORTANT EMNIST ORIENTATION FIX
    # ========================================================

    # EMNIST images used during training are stored in a
    # rotated/transposed orientation.
    #
    # Convert our normal human-written letter into the
    # orientation expected by the trained model.

    square = cv2.transpose(square)


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    square = square.astype("float32") / 255.0


    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    square = square.reshape(
        1,
        28,
        28,
        1
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        square,
        verbose=0
    )

    index = np.argmax(prediction[0])

    confidence = float(
        np.max(prediction[0]) * 100
    )


    # 0 = A
    # 1 = B
    # ...
    # 25 = Z

    letter = chr(
        ord("A") + index
    )

    return letter, confidence


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera frame error")
        continue


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )

    frame = cv2.resize(
        frame,
        (640, 480)
    )


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    results = hands.process(rgb)


    # ========================================================
    # HAND FOUND
    # ========================================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]


        # ----------------------------------------------------
        # Draw hand skeleton
        # ----------------------------------------------------

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


        # ----------------------------------------------------
        # Index fingertip
        # ----------------------------------------------------

        tip = hand.landmark[8]

        x = int(
            tip.x * 640
        )

        y = int(
            tip.y * 480
        )


        # Keep inside screen

        x = max(
            0,
            min(639, x)
        )

        y = max(
            0,
            min(479, y)
        )


        # ----------------------------------------------------
        # Fingertip circle
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (x, y),
            12,
            (0, 0, 255),
            -1
        )


        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        if previous_point is not None:

            cv2.line(
                canvas,
                previous_point,
                (x, y),
                255,
                6,
                cv2.LINE_AA
            )

        previous_point = (
            x,
            y
        )


    else:

        # Hand disappeared
        previous_point = None


    # ========================================================
    # PUT DRAWING OVER CAMERA
    # ========================================================

    drawing = cv2.cvtColor(
        canvas,
        cv2.COLOR_GRAY2BGR
    )

    output = cv2.addWeighted(
        frame,
        0.8,
        drawing,
        0.8,
        0
    )


    # ========================================================
    # RESULT BOX
    # ========================================================

    cv2.rectangle(
        output,
        (10, 10),
        (630, 110),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        output,
        "AI AIR LETTER RECOGNITION",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        "LETTER: " + detected_letter,
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )


    cv2.putText(
        output,
        f"{confidence:.1f}%",
        (250, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        output,
        "INDEX FINGER = DRAW | R = RECOGNIZE | C = CLEAR | Q = QUIT",
        (10, 465),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AI AIR LETTER RECOGNITION",
        output
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    if key == ord("c"):

        canvas = np.zeros(
            (480, 640),
            dtype=np.uint8
        )

        detected_letter = "-"

        confidence = 0.0

        print("Canvas cleared")


    # --------------------------------------------------------
    # RECOGNIZE
    # --------------------------------------------------------

    elif key == ord("r"):

        if np.count_nonzero(canvas) > 100:

            detected_letter, confidence = recognize_letter(
                canvas
            )

            print(
                f"Detected: {detected_letter} "
                f"({confidence:.1f}%)"
            )

        else:

            print("Draw a letter first!")


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

hands.close()

cv2.destroyAllWindows()

print("Program closed")