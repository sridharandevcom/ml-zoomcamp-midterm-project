import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb

# Load model + vectorizer
with open("xgb_model.bin", "rb") as f:
    dv, model = pickle.load(f)

app = FastAPI()

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Convert to dict and vectorize
    X = dv.transform([passenger.dict()])
    dX = xgb.DMatrix(X)

    # Predict
    prob = float(model.predict(dX)[0])
    pred = int(prob >= 0.5)

    return {
        "survived": pred,
        "probability_survive": prob
    }
