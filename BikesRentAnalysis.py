import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
def main():
    file_path = 'bikes_rent/bikes_rent.csv'
    data = pd.read_csv(file_path)
    data = data.drop(["season", "atemp", "windspeed(mph)"], axis = 1)
    # sns.heatmap(data=data.corr(), annot=True, cmap="coolwarm")
    # plt.show()
    # print(data)
    X, y = data.drop(["cnt"], axis = 1), data["cnt"]
    my_x, my_y = X.head(1), y.head(1)

    lr = LinearRegression()

    lr.fit(X, y)
    my_x, my_y = X.loc[[3]], y.loc[3]
    print(lr.predict(my_x))
    # print(my_y)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(lr.score(X_test, y_test))

    # print(len(X_train)/(len(X_test) + len(X_train)))
    # print(len(X))

    predict = lr.predict(X_test)
    A = np.sum((y_test-predict)**2)
    B = np.sum((y_test-y_test.mean())**2)
    print(1-A/B)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    plt.scatter(X_pca[:,0], X_pca[:,1], c = y)
    plt.show()

if __name__ == '__main__':
    main()