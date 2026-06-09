import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_images_and_labels(image_folder):
    images = []
    labels = []
    label_mapping = {'riped': 0, 'unriped': 1}  # Adjust this dictionary based on your classes
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        print(f"Processing file: {file_path}")
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to a fixed size
            images.append(img)
            label_str = filename.split('_')[0]
            if label_str in label_mapping:
                labels.append(label_mapping[label_str])
            else:
                print(f"Skipping file {file_path}: Label '{label_str}' not recognized")
        else:
            print(f"Failed to read image: {file_path}")
    return np.array(images), np.array(labels)

# Define folders
image_folder = 'D:\\cv\\Riped and Unriped Tomato Dataset\Images'

# Load data
images, labels = load_images_and_labels(image_folder)

# Check if images and labels are loaded correctly
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")

# Check if any images were loaded
if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images or labels found. Please check the folder path and file format.")

# Normalize images
images = images / 255.0

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=3)  # Adjust num_classes based on the number of your classes

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Adjust output layer based on the number of your classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the model
model.save('my_model.h5')

# Load the model
model = tf.keras.models.load_model('my_model.h5')

# Function to preprocess a single image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Load and preprocess a new image
image_path = 'D:\\cv\\Tomato_je.jpg'
img = preprocess_image(image_path)

# Make a prediction
prediction = model.predict(img)
predicted_class = np.argmax(prediction, axis=1)

if predicted_class[0]==1:
    result='unriped tomato'
elif predicted_class[0]==0:
    result='Riped tomato'


print(f'Predicted class: {predicted_class[0]}')
print('Result',result)
