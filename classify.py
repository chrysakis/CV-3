import cv2
import numpy as np
from utils import image_to_vector, vector_to_image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load training data
directory = '../data/clean/'
with open(directory + 'train.pkl', 'rb') as input:
    data = pickle.load(input)
input = data['inputs']
labels = data['labels']

# Principal components projection
pca = PCA(n_components=15)
projection = pca.fit_transform(input)

# k-nearest neighbors training
classifierKNN = KNeighborsClassifier(n_neighbors=2)
classifierKNN.fit(projection, labels)

# Support vector machine training
classifierSVM = LinearSVC()
classifierSVM.fit(projection, labels)

# Logistic regression training
classifierLR = LogisticRegression()
classifierLR.fit(projection, labels)

# Load testing data
pass

# k-nearest neighbors testing
pass

# Support vector machine testing
pass

# Logistic regression testing
pass
