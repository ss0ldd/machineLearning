import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import VarianceThreshold


df = pd.read_csv('AmesHousing (2).csv')

print("Информация о данных:")
print(df.info())

target = 'SalePrice'

X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
y = df[target]

X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = VarianceThreshold(threshold=0.1)
X_reduced = selector.fit_transform(X_scaled)
selected_features = X.columns[selector.get_support()]
X_reduced_df = pd.DataFrame(X_reduced, columns=selected_features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_reduced_df)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap='viridis', s=50)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('SalePrice')
plt.title("3D график: SalePrice от пониженных признаков")
plt.colorbar(scatter, label='SalePrice')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRMSE: {rmse:.2f}")

from sklearn.linear_model import Ridge

alphas = np.logspace(-4, 4, 100)
errors = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    errors.append(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

plt.figure(figsize=(10, 6))
plt.plot(alphas, errors, marker='.')
plt.xscale('log')
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('RMSE')
plt.title('Зависимость ошибки от alpha')
plt.grid(True)
plt.show()

from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, max_iter=10000)
lasso.fit(X_scaled, y)

coefficients = lasso.coef_
feature_names = selected_features

importance = pd.Series(coefficients, index=feature_names).abs().sort_values(ascending=False)
print("\nВажность признаков:")
print(importance)

most_important = importance.idxmax()
print(f"\nНаиболее важный признак: {most_important}")