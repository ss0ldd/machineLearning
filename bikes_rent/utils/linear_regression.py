# utils/linear_regression.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def simple_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    return model

def plot_regression_line(X, y, model):
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', label='Данные')
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    plt.plot(x_range, y_pred, color='red', label='Линия регрессии')
    plt.xlabel("Погода (weathersit)")
    plt.ylabel("Количество аренд (cnt)")
    plt.legend()
    plt.title("Простая линейная регрессия")
    plt.grid(True)
    plt.show()