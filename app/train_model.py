import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=0.1, solver="lbfgs", max_iter=5000, random_state=42))
])

model.fit(X, y)

joblib.dump(model, "app/model.pkl")

print("Modelo guardado en app/model.pkl")