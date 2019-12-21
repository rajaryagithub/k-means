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


def calc_distance (x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d


def get_k_value(data_range, end_value, start_value, wcss_value):
    # (y1 – y2)x + (x2 – x1)y + (x1y2 – x2y1) = 0
    # https://bobobobo.wordpress.com/2008/01/07/solving-linear-equations-ax-by-c-0/
    a = wcss_value[start_value - 1] - wcss_value[end_value - 2]
    b = data_range[end_value - 2] - data_range[start_value - 1]
    c1 = data_range[start_value - 1] * wcss_value[end_value - 2]
    c2 = data_range[end_value - 2] * wcss_value[start_value - 1]
    c = c1 - c2
    distance_of_points_from_line = []
    for k in range(start_value - 1, end_value - 2):
        distance_of_points_from_line.append(
            calc_distance(data_range[k], wcss_value[k], a, b, c))
    max_dist = max(distance_of_points_from_line)
    index_max_dist = distance_of_points_from_line.index(max_dist)
    no_of_clusters = index_max_dist + 1
    return no_of_clusters


def plot_straight_line_in_elbow(data_range, end_value, start_value, wcss_value):
    plt.plot(data_range, wcss_value)
    plt.plot([start_value, end_value - 1], [wcss_value[start_value - 1],
                                            wcss_value[end_value - 2]], 'ro-')
    plt.show()


def main():
    print("Python main function")

    # Create random data
    start_value = 1
    end_value = 12
    K = range(start_value, end_value)
    X, Y = make_blobs(300, cluster_std=start_value, random_state=end_value)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # Create elbow graph to show value of K
    wcss_value = getWcss(start_value, end_value, X)

    # Plot straight line between first point and last point
    plot_straight_line_in_elbow(K, end_value, start_value, wcss_value)

    # Get K value automatically
    no_of_clusters = get_k_value(K, end_value, start_value, wcss_value)
    print("Total no of clusters are = " + str(no_of_clusters))

    # Put the
    kmeans = KMeans(no_of_clusters)
    pred_y = kmeans.fit_predict(X)
    print(pred_y)
    plt.scatter(X[:, 0], X[:, 1], cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()


if __name__ == '__main__':
    main()

