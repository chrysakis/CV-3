import cv2
import numpy as np
from utils import image_to_vector, vector_to_image
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data
directory = '../data/clean/'
with open(directory + 'train.pkl', 'rb') as input:
    data = pickle.load(input)
input = data['inputs']
labels = data['labels']

# Principal components projection
pca = PCA(n_components=7)
pca.fit(input)
projection = pca.transform(input)

# k-nearest neighbors training
classifierKNN = KNeighborsClassifier(n_neighbors=2)
classifierKNN.fit(projection, labels)

# Support vector machine training
classifierSVM = LinearSVC()
classifierSVM.fit(projection, labels)

# Logistic regression training
classifierLR = LogisticRegression()
classifierLR.fit(projection, labels)

# Load and project the testing data
directory = '../data/clean/'
with open(directory + 'test.pkl', 'rb') as input:
    data = pickle.load(input)
input = data['inputs']
labels = data['labels']
projection = pca.transform(input)

# k-nearest neighbors testing
predictions = classifierKNN.predict(projection)
print(f"Accuracy: {accuracy_score(labels, predictions)}\n")

# Support vector machine testing
predictions = classifierSVM.predict(projection)
print(f"Accuracy: {accuracy_score(labels, predictions)}\n")

# Logistic regression testing
predictions = classifierLR.predict(projection)
print(f"Accuracy: {accuracy_score(labels, predictions)}\n")

