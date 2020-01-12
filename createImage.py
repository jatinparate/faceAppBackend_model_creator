import cv2
import os
import numpy as np

faces = []
labels = []
class_names = []

for item in os.listdir('images'):
    class_names.append(item)

for user in os.listdir('images'):
    for item in os.listdir('images/' + user):
        faces.append(cv2.imread('images/' + user + '/' + item, 0))
        labels.append(class_names.index(user))

face_classifier = cv2.face.LBPHFaceRecognizer_create()
face_classifier.train(faces, np.array(labels))

face_classifier.save('classifier.yml')