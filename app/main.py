from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="API de Clasificación de Cáncer de Mama",
    description="API para clasificar tumores como benignos o malignos usando Logistic Regression.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("app/model.pkl")

class TumorData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {
        "message": "API funcionando correctamente",
        "endpoints": ["/predict", "/docs"]
    }

@app.post("/predict")
def predict(data: TumorData):
    if len(data.features) != 30:
        return {
            "error": "El modelo requiere exactamente 30 variables numéricas."
        }

    input_data = np.array(data.features).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].max()

    label = "Benigno" if prediction == 1 else "Maligno"

    return {
        "prediction": int(prediction),
        "label": label,
        "confidence": round(float(probability), 4)
    }