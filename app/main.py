from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="Predictive Maintenance API")

model = joblib.load("model/model.pkl")
features = joblib.load("model/features.pkl")

try:
    scaler = joblib.load("model/scaler.pkl")
except:
    scaler = None


class PredictionInput(BaseModel):
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


label_map = {
    0: "Critical",
    1: "Warning",
    2: "Healthy"
}


@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running"}


@app.get("/ui")
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/predict")
def predict(data: PredictionInput):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[features]

    if scaler is not None:
        input_data = scaler.transform(input_df)
    else:
        input_data = input_df

    prediction = model.predict(input_data)[0]

    return {
        "prediction_class": int(prediction),
        "prediction_label": label_map[int(prediction)]
    }