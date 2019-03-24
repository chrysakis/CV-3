import cv2
import os
from utils import extract_face, image_to_vector
import pickle
import numpy as np

n = 300
inputs = np.ones((20, n**2))
labels = np.ones(20)

directory = '../data/original/train/classA/'
new_directory = '../data/clean/'
for index, file in enumerate(os.listdir(directory)):
    image = cv2.imread(directory + file)
    face = extract_face(image, n)
    vector = image_to_vector(face)
    inputs[index, :] = vector
    labels[index] = 1

directory = '../data/original/train/classB/'
for index, file in enumerate(os.listdir(directory)):
    index += 10
    image = cv2.imread(directory + file)
    face = extract_face(image, n)
    vector = image_to_vector(face)
    inputs[index, :] = vector
    labels[index] = 0

with open(new_directory + 'train.pkl', 'wb') as output:
    pickle.dump({'inputs':inputs, 'labels':labels}, output,
                pickle.HIGHEST_PROTOCOL)

inputs = np.ones((10, n**2))
labels = np.ones(10)

directory = '../data/original/test/classA/'
for index, file in enumerate(os.listdir(directory)):
    image = cv2.imread(directory + file)
    face = extract_face(image, n)
    vector = image_to_vector(face)
    inputs[index, :] = vector
    labels[index] = 1

directory = '../data/original/test/classB/'
for index, file in enumerate(os.listdir(directory)):
    index += 5
    image = cv2.imread(directory + file)
    face = extract_face(image, n)
    vector = image_to_vector(face)
    inputs[index, :] = vector
    labels[index] = 0

with open(new_directory + 'test.pkl', 'wb') as output:
    pickle.dump({'inputs':inputs, 'labels':labels}, output,
                pickle.HIGHEST_PROTOCOL)
