# Actividad 7 - Contenerización y despliegue local

## Descripción

Este proyecto implementa una solución técnica desplegable basada en un modelo de clasificación de cáncer de mama usando el dataset Breast Cancer Wisconsin.

La solución incluye:

- API backend con FastAPI.
- Modelo de machine learning con Logistic Regression.
- Frontend HTML sencillo.
- Contenedor Docker funcional.
- Endpoint de predicción `/predict`.

## Estructura

```text
Actividad7/
├── app/
│   ├── main.py
│   ├── train_model.py
│   └── requirements.txt
├── frontend/
│   └── index.html
├── Dockerfile
└── README.md