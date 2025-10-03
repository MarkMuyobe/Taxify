from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import logging

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "taxi_gb_model.joblib")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")


def ensure_model():
    """Try to load a model from disk. If not found, train a tiny fallback model
    and save it so the service is usable out-of-the-box.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        logger.info(f"Found model file at {MODEL_PATH}; attempting to load")
        try:
            m = joblib.load(MODEL_PATH)
            # if loaded model doesn't have a model_type tag, try to tag it as sklearn
            try:
                setattr(m, "_model_type", getattr(m, "_model_type", "sklearn"))
            except Exception:
                pass
            return m
        except Exception as e:
            logger.exception(f"Failed to load model at {MODEL_PATH}; will attempt to recreate fallback. Error: {e}")
            # Move the broken model out of the way so we can recreate a fallback model
            try:
                broken = MODEL_PATH + ".broken"
                os.replace(MODEL_PATH, broken)
                logger.info(f"Renamed broken model to {broken}")
            except Exception:
                logger.debug("Could not rename broken model file; continuing without removing it.")

    # Prefer loading/training a scikit-learn model if sklearn is available.
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        import numpy as np

        logger.info("No model found -- training a small fallback scikit-learn model. This may take a few seconds.")

        # Create synthetic training data that corresponds to the categorical mapping
        X = []
        y = []
        for loc in range(3):
            for tod in range(4):
                for i in range(25):
                    X.append([loc, tod])
                    base = 5.0
                    if loc == 1:  # Airport
                        base += 8.0
                    if loc == 2:  # Suburbs
                        base += 3.0
                    if tod == 0:  # Morning
                        base -= 1.0
                    if tod == 2:  # Evening
                        base += 2.0
                    if tod == 3:  # Night
                        base += 4.0
                    y.append(base + (i % 5) * 0.3)

        X = np.array(X)
        y = np.array(y)

        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        try:
            setattr(model, "_model_type", "sklearn")
        except Exception:
            pass
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Fallback scikit-learn model trained and saved to {MODEL_PATH}")
        return model
    except Exception:
        logger.warning("scikit-learn not available: falling back to deterministic rule-based predictor.")

        # Create a simple deterministic predictor with a predict(features) -> [value] API
        class RuleBasedPredictor:
            def predict(self, X):
                out = []
                for row in X:
                    loc = int(row[0])
                    tod = int(row[1])
                    base = 5.0
                    if loc == 1:
                        base += 8.0
                    elif loc == 2:
                        base += 3.0
                    if tod == 0:
                        base -= 1.0
                    elif tod == 2:
                        base += 2.0
                    elif tod == 3:
                        base += 4.0
                    out.append(base)
                return out

        model = RuleBasedPredictor()
        try:
            setattr(model, "_model_type", "rule-based")
        except Exception:
            pass
        # Save a small sentinel object so joblib.load still works if attempted later
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception:
            logger.debug("Could not save rule-based model object; continuing without saving.")
        return model


app = FastAPI(title="Taxi Wait Time API")

# Enable CORS for local frontend usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    # Either provide latitude/longitude OR a categorical `location` string.
    latitude: float | None = None
    longitude: float | None = None
    location: str | None = None
    time_of_day: str


def haversine_km(lat1, lon1, lat2, lon2):
    # simple haversine formula
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# Define rough centroids for the three categories (latitude, longitude)
LOCATION_CENTROIDS = {
    "Downtown": (40.7128, -74.0060),
    "Airport": (40.6413, -73.7781),
    "Suburbs": (40.7891, -73.1350),
}


model = None


@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        model = ensure_model()
    except Exception as exc:
        logger.exception("Failed to load or train model on startup")
        # let the app start but raise on predict if model missing


@app.get("/")
def read_root():
    return {"status": "ok", "note": "POST /predict with {location, time_of_day}"}


@app.post("/predict")
def predict(data: InputData):
    global model
    # If startup didn't load the model for any reason, attempt to load it now.
    if model is None:
        try:
            logger.info("Model missing at predict-time; attempting to load or create fallback model.")
            model = ensure_model()
        except Exception as e:
            logger.exception("Failed to load model at predict-time")
            raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    # Map categories to numbers (adjust to match training preprocessing)
    location_map = {"Downtown": 0, "Airport": 1, "Suburbs": 2}
    time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}

    # If lat/lon provided, map to nearest centroid; otherwise use provided location string
    if data.latitude is not None and data.longitude is not None:
        best = None
        best_dist = float("inf")
        for name, (latc, lonc) in LOCATION_CENTROIDS.items():
            d = haversine_km(data.latitude, data.longitude, latc, lonc)
            if d < best_dist:
                best = name
                best_dist = d
        loc = location_map.get(best, 0)
        logger.info(f"Received coords: ({data.latitude}, {data.longitude}) -> mapped to {best} ({best_dist:.2f} km)")
    else:
        loc = location_map.get(data.location or "Downtown", 0)
        logger.info(f"No coords provided, using location string: {data.location}")

    tod = time_map.get(data.time_of_day, 0)

    features = [[loc, tod]]
    pred = model.predict(features)[0]

    # Return the chosen mapping for debugging/verification
    used_location_name = None
    for name, idx in location_map.items():
        if idx == loc:
            used_location_name = name
            break

    model_type = getattr(model, "_model_type", "unknown")
    return {"predicted_wait_time": float(pred), "used_location": used_location_name, "distance_km": round(best_dist, 3) if 'best_dist' in locals() else None, "model_type": model_type}
