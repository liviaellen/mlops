from fastapi import FastAPI, HTTPException
import mlflow
import numpy as np
from pydantic import BaseModel, Field
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Initialize FastAPI app
app = FastAPI(title="Wine Classification API")

# Define input data model
class WineFeatures(BaseModel):
    features: list[float] = Field(..., min_items=13, max_items=13, description="List of 13 wine features")

# Load model from MLFlow or train a new one if not found
def load_model():
    try:
        # Set MLFlow tracking URI
        mlflow.set_tracking_uri("http://localhost:5001")

        # Create experiment if it doesn't exist
        experiment_name = "wine-classification"
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        # Try to load the model
        model = mlflow.sklearn.load_model("models:/wine-classifier/latest")
        print("Successfully loaded model from MLFlow")
        return model

    except Exception as e:
        print(f"Could not load model from MLFlow: {e}")
        print("Training a new model...")

        # Load wine dataset
        X, y = load_wine(return_X_y=True)

        # Train a new model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Log the model to MLFlow
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_param("n_estimators", 100)

            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "wine-classifier")

        print("Successfully trained and registered new model")
        return model

# Load model at startup
model = load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Wine Classification API - Livia Ellen"}

@app.post("/predict")
def predict(wine: WineFeatures):
    try:
        features = np.array(wine.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        return {
            "prediction": int(prediction),
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
