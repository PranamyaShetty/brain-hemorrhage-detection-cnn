🧠 Brain Hemorrhage Detection using CNN + ResNet

This project detects brain hemorrhage from CT scan images using a custom CNN with Residual Blocks (ResNet-inspired architecture).
It includes model training, evaluation, and a Gradio interface for real-time image-based predictions.

📂 Project Structure
├──resnet.ipynb                        # Main notebook with code
├── datasets/                          # Dataset (NormalBrain, Hemorrhage folders inside)
├── requirements.txt                   # Dependencies
└── README.md                          # Project documentation

🚀 Features
Loads CT scan dataset (Normal vs Hemorrhage).
Builds a lightweight ResNet-inspired CNN.
Trains and evaluates with metrics (Accuracy, Precision, Recall, F1).
Generates confusion matrix and classification report.
Provides a Gradio-based Web App to upload CT scans and get predictions.

🛠️ Installation
Clone this repo and install dependencies:
git clone https://github.com/PranamyaShetty/brain-hemorrhage-detection.git
cd brain-hemorrhage-detection
pip install -r requirements.txt

If you are using Google Colab:
!pip install gradio tensorflow scikit-learn opencv-python matplotlib

📊 Dataset
Organize your dataset like this:
datasets/
   ├── NormalBrain/
   │      ├── img1.png
   │      ├── img2.png
   ├── hemorrhage/
          ├── img3.png
          ├── img4.png


▶️ Usage
Run the Notebook
jupyter notebook Brain_Hemorrhage_Detection.ipynb

Prediction via Gradio
After training, launch the app:
demo.launch(share=True)

📈 Model Performance
Model: Custom ResNet-like CNN
Training: ~10 epochs
Metrics: Accuracy, Precision, Recall, F1-score
(Results may vary depending on dataset size/quality.)

🧑‍💻 Tech Stack
Python
TensorFlow / Keras
OpenCV
Matplotlib
Scikit-learn
Gradio (for UI)

📌 Future Improvements
Add Genetic Algorithm optimization for hyperparameters.
Support more classes (e.g., subtypes of hemorrhage).
Improve UI with explainability (Grad-CAM).

