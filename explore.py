import cv2
import numpy as np
from utils import image_to_vector, vector_to_image
from sklearn.decomposition import PCA
import pickle

directory = '../data/clean/'
with open(directory + 'train.pkl', 'rb') as input:
    data = pickle.load(input)
input = data['inputs']
labels = data['labels']

pca = PCA(n_components=13)
projection = pca.fit_transform(input)
reconstruction = pca.inverse_transform(projection)
print(np.cumsum(pca.singular_values_ / np.sum(pca.singular_values_)), '\n')

for i, image in enumerate(input):
    combined = np.concatenate((vector_to_image(image),
                               vector_to_image(reconstruction[i, :])), axis=1)
    cv2.imshow('', combined)
    cv2.waitKey(0)

with open(directory + 'test.pkl', 'rb') as input:
    data = pickle.load(input)
input = data['inputs']
labels = data['labels']
projection = pca.transform(input)
reconstruction = pca.inverse_transform(projection)

for i, image in enumerate(input):
    combined = np.concatenate((vector_to_image(image),
                               vector_to_image(reconstruction[i, :])), axis=1)
    cv2.imshow('', combined)
    cv2.waitKey(0)
