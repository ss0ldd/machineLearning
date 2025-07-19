from sklearn.linear_model import LassoCV

def lasso_feature_importance(X, y):
    model = LassoCV(cv=5, random_state=42)
    model.fit(X, y)

    importance = dict(zip(X.columns, abs(model.coef_)))
    most_important = max(importance, key=importance.get)

    print("Важность признаков:")
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.2f}")

    print(f"\nНаиболее важный признак: {most_important}")
    return most_important