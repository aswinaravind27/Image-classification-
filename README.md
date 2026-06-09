# Image-classification-
Opencv &amp; Tensor flow
# Tomato Ripeness Image Classifier (AI)

A deep learning computer vision application built to automate agricultural sorting by classifying tomatoes as ripe or unripe. This project features a custom Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Scikit-Learn

## 📊 Dataset Structure
The model is trained on a custom dataset organized by filename labels:
- `riped_xx.jpg` -> Class 0 (Ripe)
- `unriped_xx.jpg` -> Class 1 (Unripe)

## 🧠 Model Architecture
- **Input Shape:** 128x128x3 (RGB Images)
- **Layers:** Convolutional Layers (Conv2D), Max Pooling (MaxPooling2D), Flatten, Dense Layers with Dropout (0.5), and a Softmax activation output layer.
- **Epochs:** 20
