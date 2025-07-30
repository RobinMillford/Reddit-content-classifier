from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd

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
model_name = "None" # Default value

print("--- Initializing Production Model Loader ---")
try:
    # FINAL ROBUST LOGIC: Using the most compatible filter string.
    # This correctly identifies only the child model runs.
    all_runs = mlflow.search_runs(
        experiment_ids="0", 
        filter_string="params.model_type != ''", # This is the correct, compatible filter
        order_by=["metrics.nsfw_f1_score DESC"]
    )

    if all_runs.empty:
        raise Exception("No valid model runs found. Please ensure the training script has run successfully.")

    best_run = all_runs.iloc[0]
    best_run_id = best_run.run_id
    model_name = best_run["params.model_type"]

    print(f"✅ Found best model: '{model_name}' from run_id: {best_run_id}")
    print(f"   NSFW F1-Score: {best_run['metrics.nsfw_f1_score']:.4f}")

    # Load the Champion Model. The vectorizer is now packaged inside it.
    model_artifact_path = f"{model_name}-classifier"
    model_uri = f"runs:/{best_run_id}/{model_artifact_path}"
    
    print(f"   Loading model and vectorizer from URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Champion model loaded and ready to serve.")

except Exception as e:
    print(f"❌ Error during model loading: {e}")
    print("   Please ensure you have run the training script ('python src/train.py') successfully.")

# --- Define Prediction Endpoint ---
@app.post("/predict")
def predict(data: dict):
    if not model:
        return {"error": "Model is not loaded. Please check the backend server logs for errors."}
    
    text = data.get("text", "")
    if not text:
        return {"error": "Input text cannot be empty."}
    
    try:
        # The input must be a pandas DataFrame for the custom model wrapper
        input_df = pd.DataFrame([text])
        prediction = model.predict(input_df)
        
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
    status = "Ready" if model else "Error during startup"
    return {
        "message": "NSFW Content Classification API",
        "status": status,
        "loaded_model": model_name
    }