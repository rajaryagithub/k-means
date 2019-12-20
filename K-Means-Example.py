import math
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


def getWcss(start_range, end_range, x_value):
    wcss = []
    for i in range(start_range, end_range):
        k_means = KMeans(n_clusters=i)
        k_means.fit(x_value)
        wcss.append(k_means.inertia_)
    plt.plot(range(start_range, end_range), wcss)
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    return wcss

# This method is still not used in
def calc_distance (x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d


def main():
    print("python main function")
    start_value = 1
    end_value = 12
    X, Y = make_blobs(300, cluster_std=start_value, random_state=end_value)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # Create elbow graph to show value of K
    wcss_value = getWcss(start_value, end_value, X)

    # Put the
    kmeans = KMeans(3)
    pred_y = kmeans.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()


if __name__ == '__main__':
    main()

