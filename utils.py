import cv2


def extract_face(image):
    path = '../misc/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(path)
    n = 300
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, dsize=(n, n))
        return face


