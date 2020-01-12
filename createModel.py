import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

class_names = ['160770107541', '160770107542', '160770107543', '160770107549']

train_images = []
train_labels = []

for student in os.listdir('images'):
    for photo in os.listdir('images/' + student):
        train_images.append(
            cv2.resize(
                plt.imread('images/' + student + '/' + photo),
                (180, 180)
            )
        )
        train_labels.append(class_names.index(student))

train_images = np.array(train_images)
train_labels = np.array(train_labels)

train_images = train_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(180,180)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_image = [cv2.resize(
    plt.imread('jatin.jpg'),
    (180, 180)
)]

test_image = np.array(test_image)

print(model.predict(test_image))
