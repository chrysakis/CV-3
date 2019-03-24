import cv2
import numpy as np
import os
from utils import image_to_vector, vector_to_image
from sklearn.decomposition import PCA
import pickle

directory = '../data/clean/'
with open(directory + 'train.pkl', 'rb') as input:
    data = pickle.load(input)

input = data['inputs']
labels = data['labels']

pca = PCA(n_components=15)
projection = pca.fit_transform(input)
reconstruction = pca.inverse_transform(projection)
print(pca.singular_values_ / np.sum(pca.singular_values_), '\n')
print(1 - pca.explained_variance_ratio_)

for i, image in enumerate(input):
    combined = np.concatenate((vector_to_image(image),
                               vector_to_image(reconstruction[i, :])), axis=1)
    cv2.imshow('', combined)
    cv2.waitKey(0)
