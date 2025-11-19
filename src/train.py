import pandas as pd
import numpy as np
import os
import dagshub
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv 
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier

# --- Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
load_dotenv()

DAGSHUB_OWNER = os.getenv("DAGSHUB_OWNER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Reddit_Content_Benchmark")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Validate config
if not DAGSHUB_OWNER or not DAGSHUB_REPO or not DAGSHUB_TOKEN:
    raise ValueError("❌ Error: DAGSHUB_OWNER, DAGSHUB_REPO, or DAGSHUB_TOKEN missing in .env file.")

print("--- Starting Model Training & Automatic Selection ---")

try:
    print("🔌 Connecting to DagsHub...")
    # Force authentication using environment variables
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✅ MLflow tracking set to: {mlflow.get_tracking_uri()}")
except Exception as e:
    print(f"⚠️ Connection Warning: {e}")

# ==========================================
# 2. DATA PREPARATION
# ==========================================
print("⏳ Loading and processing data...")

if os.path.exists("data/raw_posts.csv"):
    df = pd.read_csv("data/raw_posts.csv")
else:
    # Fallback dummy data
    print("⚠️ Data file not found, generating DUMMY data.")
    df = pd.DataFrame({
        'title': ['Safe']*50, 'body': ['Safe']*50, 'classification': ['SFW']*50,
        'safety':['Safe']*50, 'toxicity':['Low']*50, 'sentiment':['Pos']*50, 
        'topic':['News']*50, 'engagement':['High']*50
    })

df['text'] = df['title'] + ' ' + df['body'].fillna('')
df['binary_label'] = df['classification'].apply(lambda label: 1 if label == 'NSFW' else 0)

label_columns = ['safety', 'toxicity', 'sentiment', 'topic', 'engagement']
label_encoders = {}
encoded_labels = []

for col in label_columns:
    unique_labels = df[col].unique()
    label_encoders[col] = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels.append(df[col].map(label_encoders[col]).values)

y_multi = np.column_stack(encoded_labels)
y_binary = df['binary_label'].values

print("⏳ Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text'])

X_train, X_test, y_bin_train, y_bin_test, y_mul_train, y_mul_test = train_test_split(
    X, y_binary, y_multi, test_size=0.2, random_state=42
)

# ==========================================
# 3. MODEL DEFINITIONS
# ==========================================
# Binary Models
binary_models = {
    "Binary_LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Binary_LinearSVC": LinearSVC(class_weight='balanced', dual="auto"),
    "Binary_MultinomialNB": MultinomialNB(),
    "Binary_LightGBM": lgb.LGBMClassifier(class_weight='balanced', verbose=-1),
    "Binary_MLP": MLPClassifier(max_iter=50, hidden_layer_sizes=(50,), early_stopping=True)
}

# Multi-Label Models
multi_models = {
    "Multi_LogisticRegression": MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight='balanced')),
    "Multi_LinearSVC": MultiOutputClassifier(LinearSVC(class_weight='balanced', dual="auto")),
    "Multi_LightGBM": MultiOutputClassifier(lgb.LGBMClassifier(class_weight='balanced', verbose=-1)),
    "Multi_MLP": MultiOutputClassifier(MLPClassifier(max_iter=50, hidden_layer_sizes=(50,), early_stopping=True)),
    "Multi_MultinomialNB": MultiOutputClassifier(MultinomialNB())
}

# ==========================================
# 4. TRAINING LOOPS
# ==========================================
best_binary_model = None
best_binary_name = ""
best_binary_score = -1 

best_multi_model = None
best_multi_name = ""
best_multi_score = -1 

# --- Train Binary ---
print("\n=== 🚀 Training Binary Models ===")
for name, model in binary_models.items():
    with mlflow.start_run(run_name=name):
        X_train_run = X_train.toarray() if "MLP" in name else X_train
        X_test_run = X_test.toarray() if "MLP" in name else X_test
        
        model.fit(X_train_run, y_bin_train)
        y_pred = model.predict(X_test_run)
        
        report = classification_report(y_bin_test, y_pred, output_dict=True)
        nsfw_f1 = report['1']['f1-score'] if '1' in report else 0
        acc = accuracy_score(y_bin_test, y_pred)
        
        mlflow.log_params(model.get_params())
        mlflow.log_metric("f1_score_nsfw", nsfw_f1)
        mlflow.log_metric("accuracy", acc)
        
        # SAFETY GUARD: Try to upload model, skip if DagsHub fails
        try:
            mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"   ⚠️ Model upload failed (API mismatch), but training continues.")

        print(f"   trained {name} | NSFW F1: {nsfw_f1:.4f}")

        if nsfw_f1 > best_binary_score:
            best_binary_score = nsfw_f1
            best_binary_model = model
            best_binary_name = name

# --- Train Multi-Label ---
print("\n=== 🚀 Training Multi-Label Models ===")
for name, model in multi_models.items():
    with mlflow.start_run(run_name=name):
        X_train_run = X_train.toarray() if "MLP" in name else X_train
        X_test_run = X_test.toarray() if "MLP" in name else X_test
        
        model.fit(X_train_run, y_mul_train)
        y_pred = model.predict(X_test_run)
        
        accuracies = [accuracy_score(y_mul_test[:, i], y_pred[:, i]) for i in range(y_mul_test.shape[1])]
        avg_acc = np.mean(accuracies)
        
        mlflow.log_metric("average_accuracy", avg_acc)
        
        # SAFETY GUARD
        try:
            mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"   ⚠️ Model upload failed (API mismatch), but training continues.")
        
        print(f"   trained {name} | Avg Acc: {avg_acc:.4f}")
        
        if avg_acc > best_multi_score:
            best_multi_score = avg_acc
            best_multi_model = model
            best_multi_name = name

# ==========================================
# 5. SAVING CHAMPIONS
# ==========================================
print("\n" + "="*40)
print(f"🥇 BEST BINARY MODEL:      {best_binary_name} (F1: {best_binary_score:.4f})")
print(f"🥇 BEST MULTI-LABEL MODEL: {best_multi_name} (Acc: {best_multi_score:.4f})")
print("="*40)

print("\n💾 Saving Best Models Locally...")
dump(vectorizer, 'tfidf_vectorizer.joblib')
metadata = {'label_columns': label_columns, 'label_encoders': label_encoders}
dump(metadata, 'model_metadata.joblib')

if best_binary_model:
    dump(best_binary_model, 'best_binary_model.joblib')
    print(f"   ✅ Saved 'best_binary_model.joblib'")

if best_multi_model:
    dump(best_multi_model, 'best_multi_model.joblib')
    print(f"   ✅ Saved 'best_multi_model.joblib'")

print("\n✅ DONE.")