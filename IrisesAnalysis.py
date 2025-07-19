import sklearn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def main():
    irises = load_iris()
    data = irises.data
    print(data[:,:1])
    target = irises.target #классы
    plt.scatter(data[:,:1], data[:,1:2], c = target)
    #plt.show()

    kmeans = KMeans(n_clusters=3)
    #kmeans.fit(data)
    #predict = kmeans.predict(data)

    predict = kmeans.fit_predict(data)
    print(predict)
    plt.scatter(data[:,:1], data[:,1:2], c = predict)
    #plt.show()
    new = [4, 4, 4, 4]
    print(k)


if __name__ == '__main__':
    main()