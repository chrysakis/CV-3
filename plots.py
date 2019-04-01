import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pickle
from sklearn.decomposition import PCA
from utils import vector_to_image, image_to_vector

# Load the data
directory = '../data/clean/'
with open(directory + 'train.pkl', 'rb') as train_file:
    data = pickle.load(train_file)
input = data['inputs']
labels = data['labels']
with open(directory + 'test.pkl', 'rb') as test_file:
    data = pickle.load(test_file)
test_input = data['inputs']

# Bar plot of explained variance ratio
pca = PCA(n_components=20)
projection = pca.fit_transform(input)
evr = np.cumsum(pca.singular_values_ / np.sum(pca.singular_values_))
fig0 = plt.figure()
ax = plt.subplot(1, 1 ,1)
x = np.arange(len(evr))
bars = (np.cumsum(np.ones(20))).astype(int)
plt.bar(x, evr, align='center')
plt.xticks(x, bars)
plt.show()
fig0.savefig('../plots/evr.png', dpi=200, bbox_inches='tight')

# Scatter-plot with dots
pca = PCA(n_components=2)
projection = pca.fit_transform(input)
reconstruction = pca.inverse_transform(projection)
fig1 = plt.figure()
plt.scatter(projection[:, 0], projection[:, 1], c=labels)
plt.title('Projection of the images in the 2D eigenspace')
plt.xlabel('Eigenface 1')
plt.ylabel('Eigenface 2')
plt.show()
fig1.savefig('../plots/scatter1.png', dpi=200, bbox_inches='tight')

# Scatter-plot with images
fig2 = plt.figure()
ax = plt.subplot(1, 1, 1)
plt.gray()
plt.scatter(projection[:, 0], projection[:, 1], c=labels)
for i, vector in enumerate(input):
    image = vector_to_image(vector)
    imagebox = OffsetImage(image, zoom=0.075)
    ab = AnnotationBbox(imagebox, (projection[i, 0], projection[i, 1]),
                        pad=0.1)
    ax.add_artist(ab)
plt.title('Projection of the images in the 2D eigenspace')
plt.xlabel('Eigenface 1')
plt.ylabel('Eigenface 2')
plt.show()
fig2.savefig('../plots/scatter2.png', dpi=1000, bbox_inches='tight')

# Reconstruction of training images
number_of_components = (1, 3, 6, 10, 14, 18, 20)
images = (1, 2, 10, 16)
fig3 = plt.figure()
for j, n in enumerate(number_of_components):
    pca = PCA(n_components=n)
    projection = pca.fit_transform(input)
    for i, image in enumerate(images):
        reconstruction = pca.inverse_transform(projection)
        ax = plt.subplot(len(images), len(number_of_components),
                         i * len(number_of_components)+ j + 1)
        plt.imshow(vector_to_image(reconstruction[images[i], :]))
        if i == 0 and n < 20:
            plt.title(f"n = {n}")
        elif i == 0 and n == 20:
            plt.title("Original")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.gray()
plt.suptitle("Reconstruction of training images", )
plt.show()
fig3.savefig('../plots/reconstruction1.png', dpi=1000,
             bbox_inches='tight')

# Reconstruction of unseen images
image = plt.imread('../data/misc/face.jpg')
noise = (255 * np.random.rand(300, 300, 3)).astype(np.uint8)
test_input = np.concatenate((test_input, image_to_vector(image),
                image_to_vector(noise)))
number_of_components = (1, 3, 6, 10, 14, 18, 20)
images = (1, 6, 10, 11)
fig4 = plt.figure()
for j, n in enumerate(number_of_components):
    pca = PCA(n_components=n)
    pca.fit(input)
    projection = pca.transform(test_input)
    for i, image in enumerate(images):
        reconstruction = pca.inverse_transform(projection)
        ax = plt.subplot(len(images), len(number_of_components),
                         i * len(number_of_components) + j + 1)
        if n != 20:
            plt.imshow(vector_to_image(reconstruction[images[i], :]))
            if i == 0 and n < 20:
                plt.title(f"n = {n}")
        else:
            plt.imshow(vector_to_image(test_input[image, :]))
            if i == 0 and n == 20:
                plt.title("Original")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.gray()
plt.suptitle("Reconstruction of unseen images", )
plt.show()
fig4.savefig('../plots/reconstruction2.png', dpi=1000,
             bbox_inches='tight')
