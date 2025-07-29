import pandas as pd
import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import os

print("--- API SCRIPT VERSION 2.0 ---") # This is our debugging marker

app = FastAPI()

# --- Add CORS Middleware ---
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dynamic Model Loading from MLflow ---
model = None
vectorizer = None
model_name = "None" # Default value

print("--- Initializing Production Model Loader ---")
try:
    # FINAL ROBUST LOGIC: Filter for runs that have the 'model_type' parameter set.
    all_runs = mlflow.search_runs(
        experiment_ids="0", 
        filter_string="params.model_type IS NOT NULL",
        order_by=["metrics.nsfw_f1_score DESC"]
    )

    if all_runs.empty:
        raise Exception("No valid model runs found in mlruns directory.")

    best_run = all_runs.iloc[0]
    best_run_id = best_run.run_id
    
    if 'tags.mlflow.parentRunId' not in best_run or pd.isna(best_run["tags.mlflow.parentRunId"]):
         raise Exception(f"Best run {best_run_id} is not a nested run and has no parent.")
         
    parent_run_id = best_run["tags.mlflow.parentRunId"]
    model_name = best_run["params.model_type"]

    print(f"✅ Found best model: '{model_name}' from run_id: {best_run_id}")
    print(f"   NSFW F1-Score: {best_run['metrics.nsfw_f1_score']:.4f}")

    # Load the Vectorizer from the Parent Run
    print(f"   Loading vectorizer from parent run: {parent_run_id}")
    vectorizer_path_local = mlflow.artifacts.download_artifacts(
        run_id=parent_run_id, 
        artifact_path="vectorizer.joblib"
    )
    vectorizer = load(vectorizer_path_local)
    print("   Vectorizer loaded successfully.")

    # Load the Champion Model from its run
    model_artifact_path = f"{model_name}-classifier"
    model_uri = f"runs:/{best_run_id}/{model_artifact_path}"
    
    print(f"   Loading model from URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Champion model loaded and ready to serve.")

except Exception as e:
    print(f"❌ Error during model loading: {e}")
    print("   Please ensure you have run the training script ('python src/train.py') successfully.")

# --- Define Prediction Endpoint ---
@app.post("/predict")
def predict(data: dict):
    if not model or not vectorizer:
        return {"error": "Model is not loaded. Please check the backend server logs for errors."}
    
    text = data.get("text", "")
    if not text:
        return {"error": "Input text cannot be empty."}
    
    try:
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)
        classification = "NSFW" if prediction[0] == 1 else "SFW"
        is_anomaly = bool(classification == "NSFW")

        return {
            "input_text": text, 
            "classification": classification,
            "is_anomaly": is_anomaly 
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": "Failed to make a prediction."}


@app.get("/")
def read_root():
    status = "Ready" if model and vectorizer else "Error during startup"
    return {
        "message": "NSFW Content Classification API",
        "status": status,
        "loaded_model": model_name
    }