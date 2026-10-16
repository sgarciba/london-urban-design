import joblib
import pandas as pd
from app.config import MODEL_PATH


def load_model():
    # 1: load the pipeline from MODEL_PATH and return it
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


def predict(model, features: dict) -> dict:
    # 2: turn the features dict into a one-row DataFrame
    #         (hint: wrap the dict in a list)
    X = pd.DataFrame([features])
    
    y_pred = model.predict(X)[0]

    return {
        "prediction": float(y_pred)
    }