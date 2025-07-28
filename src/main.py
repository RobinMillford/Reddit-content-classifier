from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from mlflow.tracking import MlflowClient
from joblib import load
import os

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

print("--- Initializing Production Model Loader ---")
try:
    client = MlflowClient()
    
    # 1. Find the best performing model run from all experiments
    # We filter out the parent runs and order by the F1 score to find the champion model.
    all_runs = mlflow.search_runs(experiment_ids="0", order_by=["metrics.nsfw_f1_score DESC"])
    child_runs = all_runs[all_runs['tags.mlflow.parentRunId'].notna()] # Filter for nested runs only

    if child_runs.empty:
        raise Exception("No child model runs found. Please run the training script.")

    best_run = child_runs.iloc[0]
    best_run_id = best_run.run_id
    parent_run_id = best_run["tags.mlflow.parentRunId"]
    model_name = best_run["params.model_type"]

    print(f"✅ Found best model: '{model_name}' from run_id: {best_run_id}")
    print(f"   NSFW F1-Score: {best_run['metrics.nsfw_f1_score']:.4f}")

    # 2. Load the Vectorizer from the Parent Run
    print(f"   Loading vectorizer from parent run: {parent_run_id}")
    vectorizer_path_local = mlflow.artifacts.download_artifacts(
        run_id=parent_run_id, 
        artifact_path="vectorizer/vectorizer.joblib"
    )
    vectorizer = load(vectorizer_path_local)
    print("   Vectorizer loaded successfully.")

    # 3. Load the Champion Model from its run
    # The model artifact path is now dynamic based on the model name
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
        # Transform the input text using the loaded vectorizer
        vectorized_text = vectorizer.transform([text])
        
        # Make a prediction with the champion model
        prediction = model.predict(vectorized_text)
        
        # The model outputs 1 for NSFW and 0 for SFW.
        classification = "NSFW" if prediction[0] == 1 else "SFW"
        is_anomaly = bool(classification == "NSFW")

        return {
            "input_text": text, 
            "classification": classification,
            "is_anomaly": is_anomaly # Kept for frontend compatibility
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
        "loaded_model": model_name if model else "None"
    }