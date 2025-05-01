# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = joblib.load('expenses_ai.joblib')

# Define the input schema
class PredictionInput(BaseModel):
    salary: float
    year: int
    month: int

@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array([[data.year, data.month, data.salary]])
    prediction = model.predict(features)
    return {"category": float(prediction[0])}
