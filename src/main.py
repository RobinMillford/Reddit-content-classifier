from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd
import os

# --- Configure MLflow to use Cloudflare R2 as a remote store ---
# These environment variables will be provided by Render's secret management.
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

BUCKET_NAME = os.getenv("BUCKET_NAME")
if not BUCKET_NAME:
    # Set a default for local testing if the env var isn't present
    print("WARNING: BUCKET_NAME environment variable not set.")
else:
    # *** THIS IS THE CRUCIAL FIX ***
    # We tell MLflow to use the R2 bucket as its backend for finding experiments and models.
    mlflow.set_tracking_uri(f"s3://{BUCKET_NAME}")

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

# --- Model Loading ---
model = None
model_name = "None"
model_stage = "None"

print("--- Initializing Production Model Loader ---")
try:
    # Find the best performing model from all runs in our experiment
    all_runs = mlflow.search_runs(
        experiment_names=["RedditContentClassifier"], 
        order_by=["metrics.nsfw_f1_score DESC"],
        max_results=1
    )

    if all_runs.empty:
        raise Exception("No model runs found in the remote artifact store.")

    best_run = all_runs.iloc[0]
    model_name = best_run["params.model_type"]
    registered_model_name = f"prod-{model_name}-classifier"

    print(f"✅ Found best model: '{model_name}' with F1-Score: {best_run['metrics.nsfw_f1_score']:.4f}")

    # Load the champion model directly from the model registry using its name.
    # MLflow will handle downloading it from your R2 bucket.
    model_uri = f"models:/{registered_model_name}/latest"
    print(f"   Loading model from URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Champion model loaded and ready to serve.")

except Exception as e:
    print(f"❌ Error during model loading: {e}")

# --- Define Prediction Endpoint ---
@app.post("/predict")
def predict(data: dict):
    if not model:
        return {"error": "Model is not loaded. Please check the backend server logs for errors."}
    
    text = data.get("text", "")
    if not text:
        return {"error": "Input text cannot be empty."}
    
    try:
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