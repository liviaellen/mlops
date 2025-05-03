from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="Reddit Content Moderation API",
    description="A machine learning service that analyzes Reddit comments and provides moderation recommendations.",
    version="1.0.0",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'Reddit Content Moderation API - Analyzes comments and provides moderation recommendations'}

class RedditComment(BaseModel):
    comment_text: str

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    model_pipeline = joblib.load("model.joblib")

@app.post('/moderate')
def moderate_comment(data: RedditComment):
    X = [data.comment_text]
    predictions = model_pipeline.predict_proba(X)
    return {
        'moderation_recommendation': 'Remove' if predictions[0][1] > 0.5 else 'Keep',
        'confidence_score': float(max(predictions[0]))
    }
