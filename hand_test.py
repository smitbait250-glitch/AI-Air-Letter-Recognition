import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

previous_x = None
previous_y = None

print("Camera started")
print("Raise your INDEX finger and move it")
print("Press C = clear")
print("Press Q = quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Index finger tip = landmark 8
        finger = hand.landmark[8]

        x = int(finger.x * 640)
        y = int(finger.y * 480)

        # Show fingertip
        cv2.circle(
            frame,
            (x, y),
            12,
            (0, 0, 255),
            -1
        )

        # Draw continuously
        if previous_x is not None:

            cv2.line(
                canvas,
                (previous_x, previous_y),
                (x, y),
                (255, 255, 255),
                8
            )

        previous_x = x
        previous_y = y

        cv2.putText(
            frame,
            "WRITING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        previous_x = None
        previous_y = None

        cv2.putText(
            frame,
            "SHOW YOUR HAND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Combine camera and drawing
    output = cv2.add(
        frame,
        canvas
    )

    cv2.putText(
        output,
        "C = CLEAR    Q = QUIT",
        (20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AIR WRITING TEST",
        output
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):

        canvas = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

        previous_x = None
        previous_y = None

    elif key == ord("q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()