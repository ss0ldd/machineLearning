import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from sklearn.datasets import load_iris

os.makedirs("kmeans_frames", exist_ok=True)

iris = load_iris()
X = iris.data[:, :2]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

def kmeans_manual(X, k, max_iterations=100, save_frames=True):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for i in range(max_iterations):
        distances = np.array([euclidean_distance(X, centroid) for centroid in centroids])
        labels = np.argmin(distances, axis=0)

        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        if save_frames:
            plt.figure(figsize=(7, 5))
            colors = ['r', 'g', 'b', 'y', 'c', 'm']
            for cluster in range(k):
                plt.scatter(X[labels == cluster, 0], X[labels == cluster, 1], s=30, color=colors[cluster])
            plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X')
            plt.title(f"Итерация {i}")
            plt.xlabel("Признак 1")
            plt.ylabel("Признак 2")
            plt.savefig(f"kmeans_frames/frame_{i:03d}.png")
            plt.close()

        if np.allclose(centroids, new_centroids):
            print(f"Сходится на итерации {i}")
            break

        centroids = new_centroids

    images = []
    for filename in sorted(os.listdir("kmeans_frames")):
        images.append(imageio.imread(os.path.join("kmeans_frames", filename)))
    imageio.mimsave('kmeans_animation.gif', images, duration=500, loop=0)

    print("GIF сохранен как 'kmeans_animation.gif'")

kmeans_manual(X, k=3)