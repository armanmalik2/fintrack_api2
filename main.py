# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Load your trained model
model, scaler = joblib.load('expenses.joblib')


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the input schema
class PredictionInput(BaseModel):
    salary: float
    year: int
    month: int

@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array([[data.year, data.month, data.salary]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return {"predicted_expenses": float(prediction[0])}
