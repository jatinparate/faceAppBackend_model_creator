import cv2
import numpy as np

face_classifier = cv2.face.LBPHFaceRecognizer_create()

face_classifier = face_classifier.load('classifier.yml')
image = cv2.imread('jatin.jpg', cv2.IMREAD_GRAYSCALE)
print(face_classifier.predict(image))