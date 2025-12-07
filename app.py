# app.py
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
from pathlib import Path
import traceback
import numpy as np
import httpx
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS - permissive for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler (keeps CORS headers)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )

# ---------- Model loading ----------
model_path = Path(__file__).parent / "model.pkl"
if not model_path.exists():
    logger.error("Model file not found at %s", model_path)
    # Fail fast so Render build logs show the reason
    raise FileNotFoundError(f"model.pkl not found at {model_path}")

try:
    model = joblib.load(model_path)  # pipeline expected
    logger.info("Loaded model from %s", model_path)
except Exception as e:
    logger.exception("Failed to load model.pkl")
    raise

# Try to get classifier step (works for Pipeline or plain estimator)
classifier = None
class_names = None

# If pipeline-like with named_steps or steps, attempt multiple safe ways:
try:
    if hasattr(model, "named_steps"):
        # Common name used by some pipelines: 'clf' or 'classifier'
        classifier = model.named_steps.get("clf") or model.named_steps.get("classifier")
        if classifier is None:
            # fallback: last step in the pipeline
            try:
                classifier = list(model.named_steps.values())[-1]
            except Exception:
                classifier = None
    elif hasattr(model, "steps"):
        classifier = model.steps[-1][1]
    else:
        # model itself may be the classifier
        classifier = model
except Exception:
    classifier = model  # last resort

# Try to get class names if present (sklearn: classes_)
if classifier is not None:
    class_names = getattr(classifier, "classes_", None)

# Fallback label names (adjust to your mapping if needed)
LABEL_NAMES = {0: "true", 1: "false"}
if class_names is not None and len(class_names) == 2:
    # If classes are strings, use them; otherwise keep fallback mapping
    if all(isinstance(c, str) for c in class_names):
        LABEL_NAMES = {i: str(class_names[i]) for i in range(len(class_names))}

# ---------- Request models ----------
class RequestModel(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sachjaano API is running"}

# News API config
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "22fd7847be494e6c97f3ce98a5ca6228")
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

@app.get("/news")
async def get_news(
    category: str = Query("general"),
    country: str = Query("us"),
    api_key: str | None = Query(default=None, alias="apiKey")
):
    key_to_use = api_key or NEWS_API_KEY
    # Only fail if key is missing or is the placeholder
    if not key_to_use or key_to_use == "YOUR_NEWS_API_KEY_HERE":
        raise HTTPException(status_code=500, detail="News API key not configured. Set NEWS_API_KEY env var.")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                NEWS_API_URL,
                params={"country": country, "category": category, "apiKey": key_to_use, "pageSize": 20},
                timeout=10.0
            )
            if response.status_code != 200:
                data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                raise HTTPException(status_code=response.status_code, detail=data.get("message", "News API error"))
            data = response.json()
            if data.get("status") == "ok":
                articles = [a for a in data.get("articles", []) if a.get("title") != "[Removed]"]
                return {"status": "ok", "totalResults": len(articles), "articles": articles}
            else:
                raise HTTPException(status_code=500, detail=data.get("message", "Failed to fetch news"))
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to news API timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to news API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {e}")

# ---------- Prediction endpoint ----------
@app.post("/predict")
def predict(req: RequestModel):
    try:
        # Use pipeline predict (works for plain estimator too)
        raw_pred = model.predict([req.text])
        if isinstance(raw_pred, (list, tuple, np.ndarray)):
            prediction = raw_pred[0]
        else:
            prediction = raw_pred

        # Prepare label and label_name safely
        label_name = None
        label_value = None

        # If prediction is string (e.g., 'fake'/'real'), return directly
        if isinstance(prediction, str):
            label_name = prediction
            label_value = prediction
        else:
            # Try numeric conversion
            try:
                label_value = int(prediction)
                label_name = LABEL_NAMES.get(label_value, str(prediction))
            except Exception:
                # fallback to stringified prediction
                label_value = str(prediction)
                label_name = str(prediction)

        # Confidence calculation: prefer model.decision_function if available,
        # otherwise try classifier.decision_function
        confidence = None
        try:
            if hasattr(model, "decision_function"):
                margin = model.decision_function([req.text])[0]
            elif classifier is not None and hasattr(classifier, "decision_function"):
                margin = classifier.decision_function([req.text])[0]
            else:
                margin = None

            if margin is not None:
                # margin can be scalar or array -> use max element for multiclass
                if hasattr(margin, "__len__"):
                    margin_val = float(max(np.abs(margin)))
                else:
                    margin_val = float(abs(margin))
                confidence = 1.0 / (1.0 + np.exp(-margin_val))
        except Exception as e:
            logger.debug("Confidence calculation failed: %s", e)
            confidence = None

        return {"label": label_value, "label_name": label_name, "confidence": confidence}
    except Exception as e:
        logger.exception("Prediction error")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Prediction error: {str(e)}", "error_type": type(e).__name__},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

# Optional local run (keeps compatibility for local dev)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    # For local testing only; Render should use its Start Command
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
