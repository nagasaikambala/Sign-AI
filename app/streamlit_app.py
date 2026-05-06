import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="Sign Language Recognizer")

st.title("🤟 Sign Language Alphabet Recognizer")

st.write("Show hand signs to detect alphabets.")

# =========================
# LOAD MODEL
# =========================

model = load_model("model/keras_model.h5", compile=False)

labels = []

with open("model/labels.txt", "r") as f:
    for line in f:
        labels.append(line.strip().split(" ")[1])

# =========================
# HAND DETECTOR
# =========================

detector = HandDetector(maxHands=1)

imgSize = 224
offset = 20

# =========================
# VIDEO PROCESSOR
# =========================

class SignLanguageProcessor(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        imgOutput = img.copy()

        hands, img = detector.findHands(img)

        if hands:

            hand = hands[0]

            x, y, w, h = hand['bbox']

            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)

            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                aspectRatio = h / w

                try:

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

                    imgInput = cv2.cvtColor(
                        imgWhite,
                        cv2.COLOR_BGR2RGB
                    )

                    imgInput = imgInput / 255.0

                    imgInput = np.expand_dims(
                        imgInput,
                        axis=0
                    )

                    prediction = model.predict(
                        imgInput,
                        verbose=0
                    )

                    index = np.argmax(prediction)

                    confidence = prediction[0][index]

                    label = labels[index]

                    cv2.rectangle(
                        imgOutput,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 255),
                        3
                    )

                    cv2.putText(
                        imgOutput,
                        f"{label} ({confidence:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                except:
                    pass

        return imgOutput

# =========================
# START STREAM
# =========================

webrtc_streamer(
    key="sign-language",
    video_transformer_factory=SignLanguageProcessor
)