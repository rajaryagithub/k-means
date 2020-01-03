import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.cluster import KMeans

# Import the library that contains our dataset
from sklearn.datasets import load_digits

# Load digit images.
# Pixels from the square image of  8×8  pixels have been reshaped in a row of  64  elements.
# Therefore, each row is an object or data.
# The characteristics or properties of each object are the gray intensities of each pixel.
# That is, we have, for each image,  64  properties.
digits = load_digits()
data = digits.data
print(data.shape)

# To improve the visualization, we invert the colors
data = 255-data

# We fix the seed to obtain the initial centroids, so the results obtained here are repeatable.
np.random.seed(1)

# Since we have 10 different digits (from 0 to 9) we choose to group the images in  10  clusters
# Classify the data with k-means
n = 10
kmeans = KMeans(n_clusters=n, init='random')
kmeans.fit(data)
Z = kmeans.predict(data)

# We plot the resulting clusters
for i in range(0, n):

    row = np.where(Z == i)[0]  # row in Z for elements of cluster i
    num = row.shape[0]       # number of elements for each cluster
    r = np.floor(num/10.)    # number of rows in the figure of the cluster

    print("cluster "+str(i))
    print(str(num)+" elements")

    plt.figure(figsize=(10,10))
    for k in range(0, num):
        plt.subplot(r+1, 10, k+1)
        image = data[row[k], ]
        image = image.reshape(8, 8)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()
