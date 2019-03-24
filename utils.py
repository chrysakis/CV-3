import cv2
import numpy as np


def extract_face(image, n):
    path = '../misc/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, dsize=(n, n))
        return face


def image_to_vector(image, color=False):
    if not color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.reshape(1, -1) / 255


def vector_to_image(vector, color=False):
    length = vector.size
    if color:
        n = round((length / 3)**0.5)
        return vector.reshape(n, n, 3)
    else:
        n = round(length**0.5)
        return vector.reshape(n, n)
