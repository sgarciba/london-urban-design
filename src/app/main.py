from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import MODEL_VERSION
from app.model import load_model, predict
from app.schemas import PredictionRequest, PredictionResponse

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model"] = load_model()   # runs once at startup
    yield
    ml_models.clear()                         # runs at shutdown


app = FastAPI(title="London Urban Temperature Predictor", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}
    


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    # TODO 2: convert the request into a dict with the ORIGINAL column names
    #         (hint: request.model_dump(...) — look up the by_alias argument)
    columns = request.model_dump(by_alias=True)

    # TODO 3: call your predict() function with the loaded model
    predictions = predict(ml_models["model"], columns)
    
    # TODO 4: return the result PLUS the model_version
    return {**predictions, "model_version": MODEL_VERSION}