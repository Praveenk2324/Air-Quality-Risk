import json
import math
import numpy as np
import torch
from kafka import KafkaConsumer
import joblib
from src.model.mlp import AirQualityMLP
from src.config import FEATURE_COLS, RISK_LABELS


TOPIC = "air-quality-readings"
BOOTSTRAP_SERVERS = ["localhost:9092"]

def load_model_and_scaler():
    scaler = joblib.load('data/processed/scaler.pkl')
    model = AirQualityMLP()
    model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu'))
    model.eval()
    return model, scaler

def score(reading: dict, model, scaler) -> dict:
    hour = reading['hour_of_day']
    month = reading['month']
    features = np.array([[
        reading['pm25'], reading["no2"], reading["o3"],
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
    ]])
    scaled = scaler.transform(features)
    x = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        probs = model.predict_proba(x)[0].numpy()
    tier = int(probs.argmax())
    return {
        "city": reading.get("city"),
        "risk_tier": tier,
        "risk_label": RISK_LABELS[tier],
        "confidence": round(float(probs[tier]), 3),
        "timestamp": reading.get("timestamp"),
    }

def consume():
    model, scaler = load_model_and_scaler()
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="air-quality-scorer-group",
    )
    print(f"Listening on topic '{TOPIC}' ...")
    for msg in consumer:
        reading = msg.value
        try:
            result = score(reading, model, scaler)
            print(f"  [{result['city']}] {result['risk_label']} "
                  f"(confidence={result['confidence']})")
        except Exception as e:
            print(f"  [ERROR] Failed to process message: {reading}. Reason: {e}")
            continue
 
 
if __name__ == "__main__":
    consume()