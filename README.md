# 🤟 Sign Language Alphabet Recognizer

An AI-powered real-time Sign Language Alphabet Recognition system built using **Deep Learning, Computer Vision, TensorFlow, OpenCV, and Streamlit**.  
The application detects hand gestures through a webcam and predicts corresponding alphabet signs in real time.

---

## 🚀 Live Demo

🔗 **Deployed Application:**  
https://sign-ai-srftqvbkwujrbsrmu2dmkp.streamlit.app/

---

## 📌 Features

- 🔤 Real-time Sign Language Alphabet Detection
- ✋ Hand Tracking using MediaPipe & CVZone
- 🧠 Deep Learning-based Gesture Classification
- 📷 Live Webcam Recognition
- ⚡ Real-time Prediction with Confidence Score
- 🌐 Streamlit Web Application Deployment
- 🎯 White-background preprocessing for improved accuracy
- 🔄 Stable prediction smoothing logic

---

## 🛠️ Tech Stack

### Languages & Frameworks
- Python
- Streamlit

### AI / ML Libraries
- TensorFlow / Keras
- NumPy
- Scikit-learn

### Computer Vision
- OpenCV
- MediaPipe
- CVZone

### Deployment
- Streamlit Cloud
- GitHub

---

## 📂 Project Structure

```bash
Sign-AI/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── train/
│
├── model/
│   ├── keras_model.h5
│   └── labels.txt
│
├── src/
│   ├── train_model.py
│   └── predict.py
│
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
🧠 Model Workflow
Hand Detection using CVZone + MediaPipe
Hand Cropping with Offset
White Background Preprocessing
Image Resizing & Normalization
Deep Learning Model Prediction
Real-time Alphabet Classification
📸 Screenshots
🔹 Live Detection

<img width="1603" height="705" alt="image" src="https://github.com/user-attachments/assets/33df16fb-8625-4784-b0be-9cfeda444f42" />


🔹 Streamlit Web App

<img width="1603" height="705" alt="image" src="https://github.com/user-attachments/assets/74be5aba-b90e-48ee-af85-1ca68dc3867b" />


⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/nagasaikambala/Sign-AI.git
cd Sign-AI
2️⃣ Create Virtual Environment
python -m venv venv

Activate Environment:

Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run Locally
Streamlit Web App
streamlit run app/streamlit_app.py
☁️ Deployment

This project is deployed using Streamlit Cloud.
Deployment Steps
Push project to GitHub
Connect repository to Streamlit Cloud
Configure Python 3.10
Deploy app
🎯 Future Improvements
🗣️ Text-to-Speech Conversion
📝 Word & Sentence Formation
🌍 Multi-language Support
📱 Mobile App Integration
🤖 Transformer-based Gesture Recognition
👨‍💻 Author

Naga Sai Kambala
GitHub: https://github.com/nagasaikambala
LinkedIn: (Add your LinkedIn link)
⭐ Support

If you found this project useful:
⭐ Star the repository
🍴 Fork the project
📢 Share with others

📜 License
This project is licensed under the MIT License.
