import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Import all the models we want to test
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb

# --- Custom Model Wrapper ---
class ClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, context, model_input):
        text_data = model_input.iloc[:, 0].tolist()
        vectorized_text = self.vectorizer.transform(text_data)
        return self.model.predict(vectorized_text)

# --- Configure MLflow for Hybrid Storage ---
# These environment variables will be provided by GitHub Actions secrets.
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

BUCKET_NAME = os.getenv("BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable not set.")

# *** THIS IS THE CRUCIAL FIX ***
# 1. The tracking URI points to the local filesystem to store metadata (run info, params, metrics).
mlflow.set_tracking_uri("file:./mlruns")
# 2. The artifact_location for the experiment points to the remote R2 bucket for large files (models).
ARTIFACT_LOCATION = f"s3://{BUCKET_NAME}"
EXPERIMENT_NAME = "RedditContentClassifier"

print(f"MLflow configured to use remote artifact store: {ARTIFACT_LOCATION}")
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Main Training Logic ---
df = pd.read_csv("data/raw_posts.csv")
df['text'] = df['title'] + ' ' + df['body'].fillna('')
df['label'] = df['classification'].apply(lambda label: 1 if label == 'NSFW' else 0)
    
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)) 
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")
print(f"NSFW posts in test set: {sum(y_test == 1)}\n")

scale_pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]

models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "LinearSVC": LinearSVC(random_state=42, class_weight='balanced', dual="auto"),
    "MultinomialNB": MultinomialNB(),
    "LightGBM": lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight_value),
    "MLPClassifier": MLPClassifier(random_state=42, max_iter=20, hidden_layer_sizes=(100, 50), early_stopping=True)
}

trained_models = {}
model_scores = {}

# Create a new experiment that will store artifacts in our R2 bucket
try:
    experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id


with mlflow.start_run(experiment_id=experiment_id, run_name="Model_Selection_Run") as parent_run:
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True) as child_run:
            print(f"--- Training and Evaluating: {model_name} ---")
            model.fit(X_train, y_train)
            
            y_pred_test = model.predict(X_test)
            report = classification_report(y_test, y_pred_test, target_names=['SFW (0)', 'NSFW (1)'], output_dict=True)
            print(classification_report(y_test, y_pred_test, target_names=['SFW (0)', 'NSFW (1)']))
            
            mlflow.log_param("model_type", model_name)
            nsfw_f1 = report['NSFW (1)']['f1-score']
            mlflow.log_metric("nsfw_f1_score", nsfw_f1)
            
            wrapped_model = ClassifierWrapper(model=model, vectorizer=vectorizer)
            mlflow.pyfunc.log_model(
                artifact_path="model", 
                python_model=wrapped_model,
                registered_model_name=f"prod-{model_name}-classifier"
            )
            
            trained_models[model_name] = model
            model_scores[model_name] = nsfw_f1

print("\n✅ All models trained and registered in the remote artifact store.")