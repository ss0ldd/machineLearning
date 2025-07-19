from utils.linear_regression import simple_linear_regression, plot_regression_line
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Загрузка данных
df = pd.read_csv('bikes_rent.csv')

X_weather = df['weathersit'].values
y_cnt = df['cnt'].values

model_weather = simple_linear_regression(X_weather, y_cnt)
plot_regression_line(X_weather, y_cnt, model_weather)

# main.py
from utils.predict import prepare_data, train_2d_model, predict_cnt
from sklearn.model_selection import train_test_split

# Подготовка данных
X_pca, y = prepare_data(df)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Обучение
pca_model = train_2d_model(X_train, y_train)

# Визуализация 2D-прогнозирования
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', alpha=0.7)
plt.colorbar(label='cnt')
plt.title("2D график: cnt от пониженных признаков")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

# Пример предсказания
input_example = df.iloc[0].drop('cnt').values.tolist()
scaler = StandardScaler().fit(df.drop(columns=['cnt']))
pca = PCA(n_components=2).fit(scaler.transform(df.drop(columns=['cnt'])))

predicted = predict_cnt(pca_model, input_example, scaler, pca)
print(f"Предсказанное количество аренд: {predicted:.0f}")

from utils.lasso_analysis import lasso_feature_importance

features = df.drop(columns=['cnt'])
target = df['cnt']

lasso_feature_importance(features, target)