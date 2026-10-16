from pydantic import BaseModel, ConfigDict
#from pydantic.alias_generators import to_pascal


class PredictionResponse(BaseModel):

    prediction: float
    model_version: str
    
    
class PredictionRequest(BaseModel):

    NDVI: float
    NDBI: float 
    MNDWI: float
    ALBEDO: float
    building_density: float
    building_volume_density: float
