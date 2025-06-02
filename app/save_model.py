import joblib
import numpy as np

model = joblib.load("app/models/best_model_production.pkl")


joblib.dump(model, "models/gesture_model_converted.pkl")