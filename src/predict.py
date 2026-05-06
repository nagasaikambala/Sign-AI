import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import math
from collections import Counter
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================

model = load_model("model/keras_model.h5", compile=False)

# =========================
# LOAD LABELS
# =========================

labels = []

with open("model/labels.txt", "r") as f:
    for line in f:
        labels.append(line.strip().split(" ")[1])

# =========================
# CAMERA + HAND DETECTOR
# =========================

cap = cv2.VideoCapture(0)

detector = HandDetector(
    maxHands=1,
    detectionCon=0.8,
    minTrackCon=0.7
)

offset = 20
imgSize = 224

# =========================
# WORD VARIABLES
# =========================

word = ""

last_added_time = 0
delay = 2.5

prediction_history = []

# =========================
# MAIN LOOP
# =========================

while True:

    success, img = cap.read()

    if not success:
        break

    # Mirror camera
    img = cv2.flip(img, 1)

    imgOutput = img.copy()

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:

        hand = hands[0]

        x, y, w, h = hand['bbox']

        # =========================
        # SAFE CROP
        # =========================

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)

        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:

            # =========================
            # CREATE WHITE BACKGROUND
            # =========================

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w

            try:

                # =========================
                # PRESERVE ASPECT RATIO
                # =========================

                if aspectRatio > 1:

                    k = imgSize / h

                    wCal = math.ceil(k * w)

                    imgResize = cv2.resize(
                        imgCrop,
                        (wCal, imgSize)
                    )

                    wGap = math.ceil((imgSize - wCal) / 2)

                    imgWhite[:, wGap:wCal + wGap] = imgResize

                else:

                    k = imgSize / w

                    hCal = math.ceil(k * h)

                    imgResize = cv2.resize(
                        imgCrop,
                        (imgSize, hCal)
                    )

                    hGap = math.ceil((imgSize - hCal) / 2)

                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # =========================
                # PREPROCESS EXACTLY LIKE TRAINING
                # =========================

                imgInput = cv2.cvtColor(
                    imgWhite,
                    cv2.COLOR_BGR2RGB
                )

                imgInput = imgInput.astype(np.float32)

                imgInput = imgInput / 255.0

                imgInput = np.expand_dims(imgInput, axis=0)

                # =========================
                # MODEL PREDICTION
                # =========================

                prediction = model.predict(
                    imgInput,
                    verbose=0
                )

                index = np.argmax(prediction)

                confidence = prediction[0][index]

                label = labels[index]

                # =========================
                # STABILIZE PREDICTIONS
                # =========================

                if confidence > 0.95:

                    prediction_history.append(label)

                if len(prediction_history) > 20:
                    prediction_history.pop(0)

                # Most common recent prediction
                if len(prediction_history) > 0:

                    most_common = Counter(
                        prediction_history
                    ).most_common(1)[0]

                    stable_label = most_common[0]

                    stable_count = most_common[1]

                    # Require stable prediction
                    if stable_count >= 15:

                        current_time = time.time()

                        # Prevent spam
                        if current_time - last_added_time > delay:

                            if (
                                len(word) == 0
                                or word[-1] != stable_label
                            ):

                                word += stable_label

                                last_added_time = current_time

                    else:
                        stable_label = "..."

                else:
                    stable_label = "..."

                # =========================
                # DRAW BOX
                # =========================

                cv2.rectangle(
                    imgOutput,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    3
                )

                # =========================
                # CURRENT LETTER
                # =========================

                cv2.putText(
                    imgOutput,
                    f"{stable_label} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # =========================
                # SHOW PROCESSED HAND
                # =========================

                cv2.imshow("Processed Hand", imgWhite)

            except:
                pass

    # =========================
    # WORD DISPLAY
    # =========================

    cv2.rectangle(
        imgOutput,
        (20, 20),
        (900, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        imgOutput,
        f"Word: {word}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        3
    )

    # =========================
    # SHOW MAIN WINDOW
    # =========================

    cv2.imshow("Sign Language", imgOutput)

    # =========================
    # KEYBOARD CONTROLS
    # =========================

    key = cv2.waitKey(1)

    # Clear word
    if key == ord('c'):
        word = ""

    # Remove last character
    if key == 8:
        word = word[:-1]

    # Exit
    if key == 27:
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()