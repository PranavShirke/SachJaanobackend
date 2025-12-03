from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
from pathlib import Path
import traceback
import numpy as np
import httpx
import os

app = FastAPI()

# Add CORS middleware to allow frontend to connect
# Allow all origins for development (more permissive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler to ensure CORS headers are always sent for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        # Let FastAPI handle HTTPExceptions normally (CORS middleware will add headers)
        raise exc
    # Handle other exceptions
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Load model - use absolute path based on this file's location
model_path = Path(__file__).parent / "model.pkl"
model = joblib.load(model_path)  # Pipeline: tfidf + LinearSVC

# Get class names from the model
# The model is a pipeline, so we need to access the classifier step
classifier = model.named_steps.get('clf') or model[-1]  # Get the classifier from pipeline
class_names = getattr(classifier, 'classes_', None)

# Create a mapping from numeric labels to class names
# Update these names based on what your model actually classifies
# Common examples: ['Negative', 'Positive'], ['Spam', 'Ham'], ['Fake', 'Real'], etc.
LABEL_NAMES = {
    0: "true",  # 0 means True (authentic news)
    1: "false",  # 1 means False (fake/misinformation)
}

# If model has classes_, try to use them
# Only overwrite if classes are already meaningful strings, not numeric
if class_names is not None and len(class_names) == 2:
    # Check if classes are strings (not numeric)
    if all(isinstance(c, str) or (hasattr(c, 'dtype') and np.issubdtype(c.dtype, np.str_)) for c in class_names):
        # If classes are already strings, use them directly
        LABEL_NAMES = {i: str(class_names[i]) for i in range(len(class_names))}
    # Otherwise, keep the LABEL_NAMES mapping we defined above

class RequestModel(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sachjaano API is running"}

# News API configuration
# Set your NewsAPI key as an environment variable: NEWS_API_KEY
# Or replace with your API key directly (not recommended for production)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY_HERE")
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

@app.get("/news")
async def get_news(
    category: str = Query("general", description="News category"),
    country: str = Query("us", description="Country code"),
    api_key: str | None = Query(
        default=None,
        description="Optional NewsAPI key override",
        alias="apiKey"
    )
):
    """
    Fetch news articles from NewsAPI.
    Categories: general, business, entertainment, health, science, sports, technology
    Countries: us, in, gb, ca, au, etc.
    """
    key_to_use = api_key or NEWS_API_KEY

    if key_to_use == "YOUR_NEWS_API_KEY_HERE" or not key_to_use:
        raise HTTPException(
            status_code=500,
            detail="News API key not configured. Please set NEWS_API_KEY environment variable."
        )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                NEWS_API_URL,
                params={
                    "country": country,
                    "category": category,
                    "apiKey": key_to_use,
                    "pageSize": 20
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_data.get("message", f"Failed to fetch news: {response.status_code}")
                )
            
            data = response.json()
            
            if data.get("status") == "ok":
                # Filter out removed articles
                articles = [article for article in data.get("articles", []) if article.get("title") != "[Removed]"]
                return {
                    "status": "ok",
                    "totalResults": len(articles),
                    "articles": articles
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=data.get("message", "Failed to fetch news")
                )
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to news API timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to news API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

@app.post("/predict")
def predict(req: RequestModel):
    try:
        # The model is a pipeline, so it should handle the text directly
        prediction = model.predict([req.text])[0]

        # LinearSVC does NOT support probabilities, but we can use decision_function
        # decision_function returns signed distance to hyperplane
        # We'll convert it to a confidence score (0-1 range) using sigmoid
        confidence = None
        if hasattr(model, "decision_function"):
            try:
                margin = model.decision_function([req.text])[0]
                if hasattr(margin, "__len__"):
                    margin = float(max(margin))
                else:
                    margin = float(margin)
                
                # Convert decision function to confidence score using sigmoid
                # This ensures confidence is always between 0 and 1
                # The absolute value ensures high confidence regardless of class
                confidence = 1.0 / (1.0 + np.exp(-abs(margin)))
            except Exception as e:
                print(f"Error calculating confidence: {e}")
                confidence = None

        # Get the class name for the prediction
        # Convert prediction to int in case it's a numpy type
        prediction_int = int(prediction)
        label_name = LABEL_NAMES.get(prediction_int, f"Unknown ({prediction})")
        
        return {
            "label": str(prediction),
            "label_name": label_name,
            "confidence": confidence
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in predict: {e}")
        print(error_details)
        # Return error with CORS headers
        return JSONResponse(
            status_code=500,
            content={"detail": f"Prediction error: {str(e)}", "error_type": type(e).__name__},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
