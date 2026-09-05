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

canvas = np.zeros((480, 640), dtype=np.uint8)

previous_point = None

print("================================")
print("      AIR WRITING")
print("================================")
print("Raise your index finger")
print("Write a letter in the air")
print("C = Clear")
print("S = Save drawing")
print("Q = Quit")

while True:

    success, frame = cap.read()

    if not success:
        print("Camera could not open")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Index finger tip
        finger = hand.landmark[8]

        x = int(finger.x * 640)
        y = int(finger.y * 480)

        # Red fingertip
        cv2.circle(
            frame,
            (x, y),
            10,
            (0, 0, 255),
            -1
        )

        # Draw
        if previous_point is not None:

            cv2.line(
                canvas,
                previous_point,
                (x, y),
                255,
                8
            )

        previous_point = (x, y)

    else:

        # Stop drawing when hand disappears
        previous_point = None

    # Put drawing over camera
    drawing = cv2.cvtColor(
        canvas,
        cv2.COLOR_GRAY2BGR
    )

    output = cv2.addWeighted(
        frame,
        0.7,
        drawing,
        0.8,
        0
    )

    cv2.putText(
        output,
        "C: CLEAR   S: SAVE   Q: QUIT",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AIR WRITING",
        output
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):

        canvas = np.zeros(
            (480, 640),
            dtype=np.uint8
        )

        previous_point = None

        print("Drawing cleared")

    elif key == ord("s"):

        cv2.imwrite(
            "drawing.png",
            canvas
        )

        print("Drawing saved as drawing.png")

    elif key == ord("q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()