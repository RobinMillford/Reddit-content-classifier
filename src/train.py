import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

# Import all the models we want to test
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb

# Start a parent MLflow run to group our experiments
with mlflow.start_run(run_name="Model Comparison and Ensemble") as parent_run:
    mlflow.log_param("parent_run", True)
    
    # --- Load and Prepare Data (Done once) ---
    df = pd.read_csv("data/raw_posts.csv")
    df['text'] = df['title'] + ' ' + df['body'].fillna('')

    # --- Create Accurate Ground Truth Labels ---
    if 'classification' not in df.columns:
        raise ValueError("The 'classification' column is missing from raw_posts.csv. Please run the latest ingest_data.py script.")
    
    df['label'] = df['classification'].apply(lambda label: 1 if label == 'NSFW' else 0)
    
    # --- Preprocessing & Feature Engineering ---
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)) 
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # --- Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")
    print(f"NSFW posts in test set: {sum(y_test == 1)}\n")

    # --- Define Models to Test ---
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

    # --- Loop Through Individual Models ---
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True) as child_run:
            print(f"--- Training and Evaluating: {model_name} ---")
            model.fit(X_train, y_train)
            print(f"{model_name} training complete.")

            y_pred_test = model.predict(X_test)
            report = classification_report(y_test, y_pred_test, target_names=['SFW (0)', 'NSFW (1)'], output_dict=True)
            print(classification_report(y_test, y_pred_test, target_names=['SFW (0)', 'NSFW (1)']))
            
            mlflow.log_param("model_type", model_name)
            nsfw_f1 = report['NSFW (1)']['f1-score']
            mlflow.log_metric("nsfw_f1_score", nsfw_f1)
            mlflow.sklearn.log_model(model, f"{model_name}-classifier")
            
            trained_models[model_name] = model
            model_scores[model_name] = nsfw_f1

    # --- Create and Evaluate Ensemble Model ---
    print("\n--- Creating and Evaluating Ensemble Model ---")
    
    top_two_models = sorted(model_scores, key=model_scores.get, reverse=True)[:2]
    print(f"Ensembling the top two models: {top_two_models[0]} and {top_two_models[1]}")

    ensemble = VotingClassifier(
        estimators=[
            (top_two_models[0], trained_models[top_two_models[0]]),
            (top_two_models[1], trained_models[top_two_models[1]])
        ],
        voting='hard'
    )

    with mlflow.start_run(run_name="VotingClassifier_Ensemble", nested=True) as child_run:
        ensemble.fit(X_train, y_train)
        print("Ensemble training complete.")
        
        y_pred_ensemble = ensemble.predict(X_test)
        report_ensemble = classification_report(y_test, y_pred_ensemble, target_names=['SFW (0)', 'NSFW (1)'], output_dict=True)
        print(classification_report(y_test, y_pred_ensemble, target_names=['SFW (0)', 'NSFW (1)']))

        mlflow.log_param("model_type", "VotingClassifier")
        mlflow.log_param("ensembled_models", top_two_models)
        mlflow.log_metric("nsfw_f1_score", report_ensemble['NSFW (1)']['f1-score'])
        mlflow.sklearn.log_model(ensemble, "ensemble-classifier")

    # Save the vectorizer once
    dump(vectorizer, 'vectorizer.joblib')
    # *** THIS IS THE FIX: Log the artifact directly without a sub-folder ***
    mlflow.log_artifact("vectorizer.joblib")

    print("\n✅ All models and ensemble trained and evaluated.")