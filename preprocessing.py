import cv2
import os
from utils import extract_face

directory = '../data/original/classA/'
new_directory = '../data/clean/classA/'
for file in os.listdir(directory):
    image = cv2.imread(directory + file)
    face = extract_face(image)
    cv2.imwrite(new_directory + file, face)

directory = '../data/original/classB/'
new_directory = '../data/clean/classB/'
for file in os.listdir(directory):
    image = cv2.imread(directory + file)
    face = extract_face(image)
    cv2.imwrite(new_directory + file, face)

