# utils/predict.py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def prepare_data(df):
    features = df.drop(columns=['cnt'])
    target = df['cnt']

    # Нормализация
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Уменьшение до 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features_scaled)

    return X_pca, target

def train_2d_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_cnt(model, input_values, scaler, pca):
    scaled = scaler.transform([input_values])
    reduced = pca.transform(scaled)
    return model.predict(reduced)[0]